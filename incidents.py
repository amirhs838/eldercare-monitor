"""Durable incident/outbox store. Single local installation, not a clinical system.

SQLite transactions atomically persist incidents and notification intent. Delivery
is at-least-once; consumers MUST deduplicate notification IDs. A transport receipt
is never human acknowledgement. Neither pose recovery nor track expiry closes an
incident. OS accounts/file ACLs are the authorization boundary for this local CLI.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

PRIORITY={'review':1,'urgent':2}


class IncidentStore:
    def __init__(self,path,ack_timeout_s=60.,reminder_s=300.):
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
               for v in (ack_timeout_s,reminder_s)):
            raise ValueError('Invalid incident deadline')
        target=Path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        # Restrictive creation before sqlite opens the database. WAL inherits mode.
        fd=os.open(str(target),os.O_CREAT|os.O_RDWR,0o600)
        os.close(fd)
        self.db=sqlite3.connect(str(target),timeout=10,isolation_level=None)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.ack_timeout_s,self.reminder_s=ack_timeout_s,reminder_s
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS incidents(
          id TEXT PRIMARY KEY, scope TEXT NOT NULL, family TEXT NOT NULL,
          priority TEXT NOT NULL, status TEXT NOT NULL,
          opened REAL NOT NULL, updated REAL NOT NULL, next_action REAL NOT NULL,
          reason TEXT NOT NULL, ack_timeout REAL NOT NULL, reminder REAL NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active ON incidents(scope,family) WHERE status!='resolved';
        CREATE TABLE IF NOT EXISTS audit(
          seq INTEGER PRIMARY KEY, incident_id TEXT, at REAL NOT NULL,
          action TEXT NOT NULL, actor TEXT NOT NULL, detail TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS outbox(
          id TEXT PRIMARY KEY, incident_id TEXT NOT NULL REFERENCES incidents(id),
          payload TEXT NOT NULL, created REAL NOT NULL, due REAL NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0, delivered REAL,
          lease_until REAL NOT NULL DEFAULT 0, lease_token TEXT, last_error TEXT);
        CREATE TABLE IF NOT EXISTS heartbeat(
          site TEXT PRIMARY KEY, at REAL NOT NULL, state TEXT NOT NULL);
        ''')

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    @staticmethod
    def _time(now):
        value=time.time() if now is None else now
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Invalid wall clock')
        return value

    def _audit(self,ident,now,action,actor,detail):
        self.db.execute('INSERT INTO audit(incident_id,at,action,actor,detail) VALUES(?,?,?,?,?)',
                        (ident,now,action,actor,detail))

    def _notify(self,row,now,action):
        mid=uuid.uuid4().hex
        payload=json.dumps({'schema_version':1,'notification_id':mid,'incident_id':row['id'],
                            'scope':row['scope'],'family':row['family'],'priority':row['priority'],
                            'status':row['status'],'reason':row['reason'],'action':action,
                            'created_at_epoch':now,'notice':'Review request, not medical diagnosis.'},allow_nan=False)
        self.db.execute('INSERT INTO outbox(id,incident_id,payload,created,due) VALUES(?,?,?,?,?)',
                        (mid,row['id'],payload,now,now))

    def get(self,ident):
        row=self.db.execute('SELECT * FROM incidents WHERE id=?',(ident,)).fetchone()
        if row is None: raise ValueError('Incident not found')
        return dict(row)

    def open(self,scope,family,priority,reason,now=None,*,new_evidence=False):
        now=self._time(now)
        if priority not in PRIORITY or not all(isinstance(v,str) and v.strip() for v in (scope,family,reason)):
            raise ValueError('Invalid incident')
        with self.transaction():
            row=self.db.execute("SELECT * FROM incidents WHERE scope=? AND family=? AND status!='resolved'",(scope,family)).fetchone()
            if row:
                ident=row['id']
                # No notification storm from repeat observations. Stronger evidence
                # reopens ACK and immediately alerts even within an existing incident.
                if PRIORITY[priority]>PRIORITY[row['priority']] or new_evidence:
                    priority = max((priority,row['priority']),key=PRIORITY.get)
                    self.db.execute("UPDATE incidents SET priority=?,status='open',updated=?,next_action=?,reason=? WHERE id=?",
                                    (priority,now,now+row['ack_timeout'],reason,ident))
                    self._audit(ident,now,'evidence_upgraded','engine',reason)
                    self._notify(self.get(ident),now,'evidence_upgraded')
                return ident
            ident=uuid.uuid4().hex
            self.db.execute('INSERT INTO incidents VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                            (ident,scope,family,priority,'open',now,now,now+self.ack_timeout_s,reason,self.ack_timeout_s,self.reminder_s))
            self._audit(ident,now,'opened','engine',reason)
            self._notify(self.get(ident),now,'opened')
            return ident

    def acknowledge(self,ident,actor,note,now=None):
        self._human_action(ident,actor,note,'acknowledged',now)

    def resolve(self,ident,actor,note,now=None):
        self._human_action(ident,actor,note,'resolved',now)

    def _human_action(self,ident,actor,note,status,now):
        now=self._time(now)
        if not isinstance(actor,str) or not actor.strip() or not isinstance(note,str) or not note.strip():
            raise ValueError('Named operator and reason are required')
        if len(actor)>128 or len(note)>2000:
            raise ValueError('Operator/note too long; do not enter unnecessary health data')
        with self.transaction():
            row=self.get(ident)
            if row['status']=='resolved': raise ValueError('Incident already resolved')
            self.db.execute('UPDATE incidents SET status=?,updated=?,next_action=? WHERE id=?',
                            (status,now,now+row['reminder'],ident))
            self._audit(ident,now,status,actor,note)
            self._notify(self.get(ident),now,status)

    def tick(self,now=None):
        """Call from independent dispatcher even while vision is unavailable."""
        now=self._time(now)
        with self.transaction():
            rows=self.db.execute("SELECT * FROM incidents WHERE status!='resolved' AND next_action<=?",(now,)).fetchall()
            for row in rows:
                action='acknowledgement_overdue' if row['status']=='open' else 'resolution_review_overdue'
                # Operational urgency, NOT newly inferred clinical deterioration.
                self.db.execute("UPDATE incidents SET priority='urgent',updated=?,next_action=? WHERE id=?",
                                (now,now+row['reminder'],row['id']))
                self._audit(row['id'],now,action,'scheduler','human_response_deadline;not_clinical_evidence')
                self._notify(self.get(row['id']),now,action)

    def heartbeat(self,site,state,now=None):
        now=self._time(now)
        self.db.execute('INSERT INTO heartbeat VALUES(?,?,?) ON CONFLICT(site) DO UPDATE SET at=excluded.at,state=excluded.state',
                        (site,now,state))

    def check_heartbeat(self,site,stale_s=30.,now=None):
        now=self._time(now)
        if not math.isfinite(stale_s) or stale_s<=0: raise ValueError('Invalid heartbeat threshold')
        row=self.db.execute('SELECT * FROM heartbeat WHERE site=?',(site,)).fetchone()
        if row is None or now-row['at']>stale_s or row['state']=='stopped':
            return self.open(f'site:{site}','service_health','urgent','vision_heartbeat_missing_or_stopped;monitoring_unavailable',now)
        return None

    def claim(self,now=None,lease_s=30.):
        now=self._time(now)
        if not math.isfinite(lease_s) or lease_s<=0: raise ValueError('Invalid lease')
        with self.transaction():
            # Preserve creation order per incident; no stale opening after a close.
            row=self.db.execute('''SELECT o.* FROM outbox o WHERE delivered IS NULL AND due<=? AND lease_until<=?
             AND NOT EXISTS(SELECT 1 FROM outbox p WHERE p.incident_id=o.incident_id AND p.delivered IS NULL AND p.rowid<o.rowid)
             ORDER BY o.rowid LIMIT 1''',(now,now)).fetchone()
            if row is None: return None
            token=uuid.uuid4().hex
            self.db.execute('UPDATE outbox SET lease_until=?,lease_token=? WHERE id=?',(now+lease_s,token,row['id']))
            return {**dict(row),'lease_token':token}

    def delivery_result(self,ident,token,success,error='',now=None):
        now=self._time(now)
        with self.transaction():
            row=self.db.execute('SELECT * FROM outbox WHERE id=? AND lease_token=? AND delivered IS NULL',(ident,token)).fetchone()
            if row is None: return False
            if success:
                self.db.execute('UPDATE outbox SET delivered=?,attempts=attempts+1,lease_until=0,lease_token=NULL,last_error=NULL WHERE id=?',(now,ident))
            else:
                delay=min(300.,2.**min(row['attempts']+1,8))
                self.db.execute('UPDATE outbox SET attempts=attempts+1,due=?,lease_until=0,lease_token=NULL,last_error=? WHERE id=?',
                                (now+delay,error[:80],ident))
            return True

    def active(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM incidents WHERE status!='resolved' ORDER BY opened")]

    def delivery_status(self):
        return dict(self.db.execute('SELECT COUNT(*) AS pending,COALESCE(MAX(attempts),0) AS max_attempts FROM outbox WHERE delivered IS NULL').fetchone())

    def close(self):
        self.db.close()

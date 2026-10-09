import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery import HTTPSender,NoRedirects,deliver_one
from incidents import IncidentStore


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'events.sqlite3'
        self.store=IncidentStore(self.path,ack_timeout_s=10,reminder_s=30)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def event(self,priority='review',scope='room:run:track-1',**kw):
        return self.store.open(scope,'safety',priority,'test_evidence',now=100,**kw)

    def test_dedup_and_atomic_outbox(self):
        first=self.event()
        self.assertEqual(self.event(),first)
        self.assertEqual(len(self.store.active()),1)
        self.assertEqual(self.store.delivery_status()['pending'],1)

    def test_rollback_cannot_leave_incident_without_notification(self):
        with patch.object(self.store,'_notify',side_effect=OSError('disk error')):
            with self.assertRaises(OSError):self.event()
        self.assertEqual(self.store.active(),[])
        self.assertEqual(self.store.delivery_status()['pending'],0)

    def test_survives_process_restart(self):
        ident=self.event('urgent')
        self.store.close()
        self.store=IncidentStore(self.path)
        self.assertEqual(self.store.get(ident)['status'],'open')
        self.assertEqual(self.store.delivery_status()['pending'],1)
        self.assertEqual(self.store.get(ident)['ack_timeout'],10)

    def test_run_scoping_does_not_guess_identity(self):
        a=self.event(scope='room:runA:track-1')
        b=self.event(scope='room:runB:track-1')
        self.assertNotEqual(a,b)
        self.assertEqual(len(self.store.active()),2)

    def test_priority_upgrade_reopens_acknowledgement(self):
        ident=self.event()
        self.store.acknowledge(ident,'operator','checking',101)
        self.store.open('room:run:track-1','safety','urgent','possible_fall',102)
        self.assertEqual(self.store.get(ident)['status'],'open')
        self.assertEqual(self.store.get(ident)['priority'],'urgent')

    def test_new_fall_same_priority_not_swallowed_by_prior_ack(self):
        ident=self.event('urgent')
        self.store.acknowledge(ident,'operator','checking',101)
        self.store.open('room:run:track-1','safety','urgent','new_fall',102,new_evidence=True)
        self.assertEqual(self.store.get(ident)['status'],'open')
        self.assertEqual(self.store.delivery_status()['pending'],3)

    def test_ack_is_not_resolution(self):
        ident=self.event()
        self.store.acknowledge(ident,'operator','on the way',101)
        self.assertEqual(len(self.store.active()),1)
        self.assertEqual(self.store.get(ident)['status'],'acknowledged')

    def test_deadline_escalation_is_not_clinical_evidence(self):
        ident=self.event()
        self.store.tick(109.9)
        self.assertEqual(self.store.get(ident)['priority'],'review')
        self.store.tick(110.)
        self.assertEqual(self.store.get(ident)['priority'],'urgent')
        self.assertEqual(self.store.delivery_status()['pending'],2)
        self.store.tick(111.)
        self.assertEqual(self.store.delivery_status()['pending'],2)
        row=self.store.db.execute('SELECT detail FROM audit ORDER BY seq DESC LIMIT 1').fetchone()
        self.assertIn('not_clinical_evidence',row['detail'])

    def test_acknowledged_incident_has_resolution_reminder(self):
        ident=self.event()
        self.store.acknowledge(ident,'operator','checking',101)
        self.store.tick(130.)
        self.assertEqual(self.store.delivery_status()['pending'],2)
        self.store.tick(131.)
        self.assertEqual(self.store.get(ident)['status'],'acknowledged')
        self.assertEqual(self.store.delivery_status()['pending'],3)

    def test_explicit_resolution_and_no_future_escalation(self):
        ident=self.event()
        self.store.resolve(ident,'operator','review completed',101)
        self.store.tick(10000.)
        self.assertEqual(self.store.active(),[])
        self.assertEqual(self.store.delivery_status()['pending'],2)
        self.assertNotEqual(self.event(),ident)

    def test_anonymous_ack_and_empty_resolution_rejected(self):
        ident=self.event()
        with self.assertRaises(ValueError): self.store.acknowledge(ident,'','checking',101)
        with self.assertRaises(ValueError): self.store.resolve(ident,'operator','',101)
        self.assertEqual(self.store.get(ident)['status'],'open')

    def test_claim_leases_prevent_two_workers_sending_same_item(self):
        self.event()
        message=self.store.claim(100.)
        self.assertIsNotNone(message)
        other=IncidentStore(self.path)
        try:self.assertIsNone(other.claim(101.))
        finally:other.close()
        renewed=self.store.claim(131.)
        self.assertEqual(renewed['id'],message['id'])
        self.assertNotEqual(renewed['lease_token'],message['lease_token'])
        self.assertFalse(self.store.delivery_result(message['id'],message['lease_token'],True,now=132))

    def test_failed_delivery_retries_same_id_with_backoff(self):
        self.event()
        class Fail:
            def send(self,message):raise TimeoutError('secret URL must not be persisted')
        self.assertTrue(deliver_one(self.store,Fail(),100.))
        self.assertIsNone(self.store.claim(101.))
        message=self.store.claim(102.)
        self.assertEqual(message['last_error'],'TimeoutError')
        self.assertEqual(message['attempts'],1)
        self.assertEqual(json.loads(message['payload'])['notification_id'],message['id'])

    def test_transport_success_not_human_ack(self):
        ident=self.event()
        class OK:
            def send(self,message):return True
        deliver_one(self.store,OK(),100.)
        self.assertEqual(self.store.delivery_status()['pending'],0)
        self.assertEqual(self.store.get(ident)['status'],'open')

    def test_per_incident_delivery_order(self):
        ident=self.event()
        self.store.resolve(ident,'operator','checked',101)
        first=self.store.claim(102.)
        self.assertEqual(json.loads(first['payload'])['action'],'opened')
        self.assertIsNone(self.store.claim(103.))
        self.store.delivery_result(first['id'],first['lease_token'],True,now=103.)
        last=self.store.claim(104.)
        self.assertEqual(json.loads(last['payload'])['action'],'resolved')

    def test_stale_heartbeat_opens_durable_health_incident(self):
        self.store.heartbeat('room','ok',100)
        self.assertIsNone(self.store.check_heartbeat('room',30,129))
        ident=self.store.check_heartbeat('room',30,131)
        self.assertEqual(self.store.get(ident)['family'],'service_health')
        self.store.heartbeat('room','ok',132)
        self.assertIsNone(self.store.check_heartbeat('room',30,132))
        self.assertEqual(self.store.get(ident)['status'],'open')

    def test_missing_and_stopped_heartbeat_alert(self):
        self.assertIsNotNone(self.store.check_heartbeat('room',now=100))
        self.store.heartbeat('other','stopped',100)
        self.assertIsNotNone(self.store.check_heartbeat('other',now=101))

    @unittest.skipIf(os.name=='nt','POSIX permissions only')
    def test_new_store_has_restrictive_permissions(self):
        self.assertEqual(self.path.stat().st_mode & 0o777,0o600)

    def test_https_and_no_credentials_in_url(self):
        for url in ('http://example.com','https://u:p@example.com','https://example.com#fragment','https://example.com?key=secret'):
            with self.assertRaises(ValueError):HTTPSender(url,'token')
        with self.assertRaises(ValueError):HTTPSender('https://example.com','')
        self.assertIsNotNone(HTTPSender('https://example.com/events','token'))
        self.assertIsNone(NoRedirects().redirect_request(None,None,302,'',{},'https://other.test'))

    def test_invalid_clock_and_deadline_rejected(self):
        with self.assertRaises(ValueError):self.store.tick(float('nan'))
        with self.assertRaises(ValueError):IncidentStore(self.path,ack_timeout_s=True)

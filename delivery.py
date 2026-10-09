#!/usr/bin/env python3
"""Independent durable HTTPS notification dispatcher and vision heartbeat watchdog.

Run as a separate supervised process. Endpoint must authenticate Bearer tokens,
validate events and deduplicate Idempotency-Key. A 2xx is only a transport receipt.
No SMS, phone call or human acknowledgement is simulated by this implementation.
"""
import argparse
import json
import math
import os
import sys
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from incidents import IncidentStore


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None  # never forward credentials through a redirect


class HTTPSender:
    def __init__(self,url,token):
        parts=urlsplit(url)
        if (parts.scheme!='https' or not parts.hostname or parts.username or parts.password
                or parts.fragment or parts.query):
            raise ValueError('Use an approved HTTPS endpoint without URL credentials/query/fragment')
        if not token or any(c in token for c in '\r\n'):
            raise ValueError('A Bearer token from the environment is required')
        self.url,self.token=url,token
        self.opener=build_opener(NoRedirects())

    def send(self,message):
        req=Request(self.url,data=message['payload'].encode('utf-8'),method='POST',headers={
            'Content-Type':'application/json','Authorization':f'Bearer {self.token}',
            'Idempotency-Key':message['id']})
        with self.opener.open(req,timeout=8) as response:
            return 200<=response.status<300


def deliver_one(store,sender,now=None):
    message=store.claim(now,lease_s=30.)
    if message is None: return False
    error=''
    try:
        success=bool(sender.send(message))
        if not success: error='NonSuccessResponse'
    except Exception as exc:
        success=False
        error=type(exc).__name__  # never log URL, secret or response body
    store.delivery_result(message['id'],message['lease_token'],success,error,now)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True)
    p.add_argument('--watch-site',help='Must match live monitor --site-id; do not watch offline replay')
    p.add_argument('--stale-seconds',type=float,default=30.)
    p.add_argument('--interval',type=float,default=1.)
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if any(not math.isfinite(v) or v<=0 for v in (args.interval,args.stale_seconds)):
        p.error('Intervals must be finite and positive')
    sender=None
    if os.environ.get('ELDERCARE_WEBHOOK_URL'):
        sender=HTTPSender(os.environ['ELDERCARE_WEBHOOK_URL'],os.environ.get('ELDERCARE_WEBHOOK_TOKEN',''))
    else:
        print('NO DELIVERY CONFIGURED: incidents remain queued locally; no caregiver is notified.',file=sys.stderr)
    store=IncidentStore(args.db)
    try:
        while True:
            if args.watch_site:
                store.check_heartbeat(args.watch_site,args.stale_seconds)
            store.tick()
            if sender:
                # Bound work so the watchdog cannot be starved by a large backlog.
                deliver_one(store,sender)
            if args.once:
                print(json.dumps(store.delivery_status()))
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'Dispatcher stopped: {type(exc).__name__}; check protected configuration.',file=sys.stderr)
        sys.exit(2)

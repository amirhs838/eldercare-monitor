#!/usr/bin/env python3
"""Local operator interface. Restrict OS access; this is NOT a remote auth server."""
import argparse
import json
import sys
from incidents import IncidentStore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    sub.add_parser('delivery-status')
    for command in ('ack','resolve'):
        q=sub.add_parser(command)
        q.add_argument('incident_id')
        q.add_argument('--operator',required=True)
        q.add_argument('--note',required=True,help='Non-sensitive reason/outcome; required audit entry')
    args=p.parse_args()
    store=IncidentStore(args.db)
    try:
        if args.command=='list':
            print(json.dumps(store.active(),ensure_ascii=False,indent=2))
        elif args.command=='delivery-status':
            print(json.dumps(store.delivery_status(),indent=2))
        else:
            operation=store.acknowledge if args.command=='ack' else store.resolve
            try:
                operation(args.incident_id,args.operator,args.note)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                sys.exit(2)
            print('Recorded. Acknowledgement is not resolution; visual movement is not medical clearance.')
    finally:
        store.close()


if __name__=='__main__': main()

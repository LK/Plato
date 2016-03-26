#!/usr/local/bin/python3

import argparse
import time
import os
import glob
import requests

def main():
    parser = argparse.ArgumentParser(description='Automatically upload recorded matches to Plato server.')
    parser.add_argument('--dir', type=str, dest='dir', default='~/robocode/robots/lk/Robot.data/', help='the directory of the recorded matches (generally the robot\'s data directory)')
    parser.add_argument('--endpoint', type=str, dest='endpoint', default='http://127.0.0.1:8080/upload', help='the endpoint to upload match results to')
    parser.add_argument('--interval', type=int, dest='interval', default=1, help='how many minutes to wait between successive uploads')

    args = parser.parse_args()

    while True:
        print('Looking for new matches to upload...', end='')
        files = []
        newest = None
        newest_time = 0
        for f in os.listdir(args.dir):
            if os.path.isfile(os.path.join(args.dir, f)) and os.path.splitext(f)[1] == '.txt':
                if newest_time < os.path.getmtime(os.path.join(args.dir, f)):
                    newest_time = os.path.getmtime(os.path.join(args.dir, f))
                    newest = f
                files.append(f)

        # Ignore the newest file in case its still being written
        # NOTE: This won't scale to multiple simultaneous battle runners
        files.remove(newest)

        if len(files) > 0:
            print('will upload %d files' % len(files))
        else:
            print('no new matches detected')

        body = {}
        for f in files:
            body[f] = open(os.path.join(args.dir, f), 'rb')

        r = requests.post(args.endpoint, files=body)
        if r.status_code != 200:
            print('[!] Received %d response' % r.status_code)
        else:
             for f in files:
                 os.remove(os.path.join(args.dir, f))

        time.sleep(60 * args.interval)

if __name__ == '__main__':
    main()

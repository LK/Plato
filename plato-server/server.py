import tornado.httpserver, tornado.ioloop, tornado.web
import os
import glob
from pymongo import MongoClient

games = None
games_history = None

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            ('/download', DownloadHandler),
            ('/upload', UploadHandler)
        ]
        tornado.web.Application.__init__(self, handlers)

class DownloadHandler(tornado.web.RequestHandler):
    def get(self):
        # http://stackoverflow.com/questions/18279063/python-find-newest-file-with-mp3-extension-in-directory
        newest = max(glob.iglob('snapshots/*.h5'), key=os.path.getctime)
        f = open(newest, 'rb')
        self.write(f.read())

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        if games == None or games_history == None:
            print('[!] Ignoring upload because database has not been initalized')
            return

        for filename, array in self.request.files.items():
            f = array[0]

            if len(f.body) < 5:
                continue

            if games.find_one({'name': os.path.splitext(filename)[0]}) != None:
                print('[!] Skipping %s; already uploaded' % filename)
                continue

            history = []
            lines = f.body.decode('utf-8').splitlines()
            if lines[-1] != 'W' and lines[-1] != 'L':
                lines.append('L')

            result = lines.pop()

            states = [state.split(' ') for state in lines]

            bad_data = False
            for state in states:
                if len(state) != 5:
                    bad_data = True
                    break

            if bad_data:
                continue

            for state in lines[:-1]:
                nums = state.split(' ')

                history.append({'heading': float(nums[0]), 'energy': float(nums[1]), 'oppBearing': float(nums[2]), 'oppEnergy': float(nums[3]), 'action': int(nums[4])})

            doc = {'name': os.path.splitext(filename)[0], 'history': history, 'res': lines[-1]}

            games.insert_one(doc)
            games_history.insert_one(doc)

def main():
    global games, games_history
    games = MongoClient('localhost', 27017).plato.games
    games_history = MongoClient('localhost', 27017).plato.games_history

    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(8080)
    print('Starting HTTP server on port 8080')
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()

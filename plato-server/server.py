import tornado.httpserver, tornado.ioloop, tornado.web

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/upload", UploadHandler)
        ]
        tornado.web.Application.__init__(self, handlers)

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        print self.request.files

def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(8080)
    print 'Starting HTTP server on port 8080'
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()

__author__ = 'pravesh'

import web
from web.contrib.template import render_jinja

urls = (
        '/', 'index',
        '/(.*)', 'pages'
)

app = web.application(urls, globals())

render = render_jinja(
        'templates',   # Set template directory.
        encoding = 'utf-8',                         # Encoding.
)


class index:
    def GET(self):
        return render.about()

class pages:
    def GET(self, page_name):
        try:
            return render.__getattr__(page_name)()
        except:
            return "404, page %s not found" % page_name


if __name__ == "__main__":
    app.run()
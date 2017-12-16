__author__ = 'pravesh'

import web
from web.contrib.template import render_jinja

urls = (
        '/', 'index'
)

app = web.application(urls, globals())

render = render_jinja(
        'templates',   # Set template directory.
        encoding = 'utf-8',                         # Encoding.
)


class index:
    def GET(self):
        return render.index()


if __name__ == "__main__":
    app.run()
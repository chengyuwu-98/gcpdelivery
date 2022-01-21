import sys
sys.path.insert(0, "/home/wcy505823098/.delivery/lib/python3.7/site-packages")

from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Welcome to the web!'

@app.route('/name/<value>')
def name(value):
    """Return a friendly HTTP greeting."""
    return 'Welcome to the web,%s!' % value

@app.route('/gender/<value>')
def gender(value):
    """Return a friendly HTTP greeting."""
    return 'Welcome to the web, %s!' % value

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return '<h1>Welcome to the web!<h1>\
    <img src="https://bluelightliving.com/wp-content/uploads/2020/09/Duke-University-Duke-Chapel-1-1-1080x675.jpg">'

@app.route('/name/<value>')
def name(value):
    """Return a friendly HTTP greeting."""
    return 'Welcome to the web, %s!' % value

@app.route('/gender/<value>')
def gender(value):
    """Return a friendly HTTP greeting."""
    return 'Welcome to the web, %s!' % value

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

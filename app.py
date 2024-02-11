from flask import Flask
try:
    import fcntl
except ImportError:
    fcntl = None

app = Flask(__name__)



@app.route('/')

def hello_world():

    return 'Hello, World!'
from flask import Flask, jsonify, render_template, request
from recognizer import *

# set the project root directory as the static folder, you can set others.
app = Flask(__name__)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/hw')
def hello_world():
    return 'Hello, World!'


@app.route('/recognizeDigit', methods=['POST'])
def recognize():
    image = request.json['image']
    digit = recognize_digit(image)
    return jsonify(int(digit))

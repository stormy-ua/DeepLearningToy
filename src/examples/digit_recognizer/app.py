from flask import Flask, jsonify, render_template, request
from recognizer import *
import numpy as np
import json

# set the project root directory as the static folder, you can set others.
app = Flask(__name__)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/recognizeDigit', methods=['POST'])
def recognize():
    image = request.json['image']
    probs = recognize_digit(image)
    return jsonify({"digit": int(np.argmax(probs)), "probs": json.dumps(probs.tolist())})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

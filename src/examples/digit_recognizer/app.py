from flask import Flask, jsonify, render_template, request
from recognizer import *
import numpy as np
import json
from pydeeptoy.io_utils.json_io import JsonSerializer

# set the project root directory as the static folder, you can set others.
app = Flask(__name__)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/comp_graph.html')
def comp_graph():
    return app.send_static_file('comp_graph.html')


@app.route('/recognizeDigit', methods=['POST'])
def recognize():
    image = request.json['image']
    probs = recognize_digit(image)
    return jsonify({"digit": int(np.argmax(probs)), "probs": json.dumps(probs.tolist())})


@app.route('/comp_graph_data')
def comp_graph_data():
    cg = ComputationalGraph()
    x_in = cg.constant(name="X.T")
    nn_output = softmax(cg, neural_network(cg, x_in, 10, 10, 10, 10, 3), name="nn_output")

    y_train = cg.constant(name="one_hot_y")
    loss = cross_entropy(cg, nn_output, y_train, "loss_cross_entropy")
    serializer = JsonSerializer()
    json = serializer.serialize(cg)

    return json


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

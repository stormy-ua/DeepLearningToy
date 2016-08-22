# Deep Learning Toy

![alt text](../master/logo.jpg)

> Lightweight deep learning library implemented in Python. Designed for studying how contemporary deep learning libraries are implemented.

[![Build Status](https://travis-ci.org/stormy-ua/DeepLearningToy.svg?branch=master)](https://travis-ci.org/stormy-ua/DeepLearningToy)
[![PyPI](https://img.shields.io/pypi/v/nine.svg?maxAge=2592000)](https://pypi.python.org/pypi/pydeeptoy)

## Architecture
There are several core ideas used by the framework: [computational graph](http://colah.github.io/posts/2015-08-Backprop/), forward propagation, [loss/cost function](https://en.wikipedia.org/wiki/Loss_function), [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and [backward propagation](http://neuralnetworksanddeeplearning.com/chap2.html).
*Computational graph* is a graph representing ordered set of primitive algeabric operations. Forward propagation feeds an input into a computational graph and produces the output. *Loss function* is a metric measuring how well a model estimates class or a value based on the input; usually, a loss function produces a scalar value. *Gradient descent* is the calculus approach for a loss function minimization. It uses the simple idea that in order to minimize a function we have to follow a path directed by its variables gradients. *Backward propagation* takes a graph in the state after forward propagation had finished, and calculates gradients starting from the output towards the input; this direction from the head of the computational graph towards the tail is the result of the calculus chain rule.

## Computational Graph
[ComputationalGraph](../master/src/pydeeptoy/computational_graph.py) class is equipped with methods representing primitive algeabric operations. Each method takes an input and produces an output. Inputs and outputs are represented by the [Connection](../master/src/pydeeptoy/nodes.py) class, and operations by the [Node](../master/src/pydeeptoy/nodes.py) class. There are two types of connections: constants and variables. The former do not change during the model optimization, but the latter could be changed during the optimization process. Here is the example of the primitive computational graph which adds two numbers:

```python
from pydeeptoy.computational_graph import *

cg = ComputationalGraph()
sum_result = cg.sum(cg.constant(1), cg.constant(2))
```
The code listed above builds the computational graph, but doesn't execute it. In order to execute the graph the [SimulationContext](../master/src/pydeeptoy/simulation.py) class should be used. The simulation context has the logic for doing forward/backward propagation. In addition, it stores all computation results produced by each and every operation, including gradients obtained during the backward phase. The code executing the computational graph described above:

```python
from pydeeptoy.computational_graph import *
from pydeeptoy.simulation import *

cg = ComputationalGraph()
sum_result = cg.sum(cg.constant(1), cg.constant(2))

ctx = SimulationContext()
ctx.forward(cg)

print("1+2={}".format(ctx[sum_result].value))
```

## Atomic Operations
A computational graph is composed from a set of operations. An operation is the minimum building block of a computational graph. In the framework an operation is represented by the abstract [Node](../master/src/pydeeptoy/nodes.py) class. All operation take an input in the form of a numpy array or a scalar value and produce either a scalar value or a numpy array. In other words, a computational graph passes a [tensor](https://en.wikipedia.org/wiki/Tensor) through itself. That is why one of the most popular deep learning framework is called [TensorFlow](https://www.tensorflow.org). The following operations are implemented in the [computational_graph](../master/src/pydeeptoy/computational_graph.py) module:

| Operation | Description
--- | ---
sum | Computes the sum of two tensors.
multiply | Computes the product of two tensors.
matrix_multiply | Computes the product of two matrices (aka 2 dimensional tensors).
div | Divides one tensor by another.
exp | Calculate the exponential of all elements in the input tensor. |
log | Natural logarithm, element-wise. |
reduce_sum | Computes the sum of elements across dimensions of a tensor. |
max | Element-wise maximum of tensor elements. |
broadcast | |
transpose | Permute the dimensions of a tensor. |
reshape | Gives a new shape to an array without changing its data. |
conv2d | Computes a 2-D convolution given 4-D input and filter tensors. |


## Activation Functions
[Activation functions](https://en.wikipedia.org/wiki/Activation_function) are used for thresholding a single neuron output. First, a neuron calculates its output based on the weighted sum of its inputs. Second, the calculated weighted sum is fed into the activation function. Finally, the activation function produces the final neuron output. Usually, an activation function ouput is normalized to be in between 0 and 1, or -1 and 1. The list of implemented activation functions:

* [Relu](../master/src/pydeeptoy/activations.py)

## Loss Functions
Loss functions are used as a mesure of the model performance. Usually, it is just a scalar value telling how well a model estimates output based on the input. Needless to say, a universal loss function which fits all model flavours doesn't exists. The following loss functions are implemented in the [losses](../master/src/pydeeptoy/losses.py) module:

* [Softmax](https://en.wikipedia.org/wiki/Softmax_function)
* [Hinge](https://en.wikipedia.org/wiki/Hinge_loss)

## Computational Graph Visualization

It is important to be able to visualize a complex computational graph. First, it helps to understand how a model works. Second, having a computational graph in the form of a visualization might help with debugging and finding an issue.

[The example](../master/src/examples/visualization/) is demonstrating how to render a computational graph on the web page using [d3.js](http://d3js.org/) library on the frontend and [Flask](http://flask.pocoo.org) web framework on the backend. Interactive demo is [here](http://quantumtunnel.xyz/comp_graph.html). It renders computational graph of the 4-layer neural network:

![alt text](../master/comp_graph_visualization.png)

## Usage Examples

> The set of primitive building blocks provided by the framework could be used to build robust estimators. The benefit of using the framework is that you do not have to implement forward/backward propagation from scratch for every kind of an estimator.

| | Iris | MNIST | CIFAR-10
--- | --- | --- | ---
| Support Vector Machine (SVM)| [Example](../master/src/examples/iris_svm_classification.ipynb) | |
| Multilayer Perceptron | [Example](../master/src/examples/iris_2_layer_neural_network_classification.ipynb) | [Example](../master/src/examples/MNIST-multilayer-perceptron.ipynb) |

## License

[MIT license](http://opensource.org/licenses/mit-license.php)

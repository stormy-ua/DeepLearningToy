# Deep Learning Toy
> Lightweight deep learning library implemented in Python. Designed for studying how contemporary deep learning libraries are implemented.

## Architecture
There are several core ideas used by the framework: [computational graph](http://colah.github.io/posts/2015-08-Backprop/), forward propagation, [loss/cost function](https://en.wikipedia.org/wiki/Loss_function), [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and [backward propagation](http://neuralnetworksanddeeplearning.com/chap2.html).
*Computational graph* is a graph representing ordered set of primitive algeabric operations. Forward propagation feeds an input into a computational graph and produces the output. *Loss function* is a metric measuring how well a model estimates class or value based on the input; usually, loss function produces a scalar value. *Gradient descent* is the calculus approach for a loss function minimization. It uses the simple idea that in order to minimize a function we have to follow a path directed by its variables gradients. *Backward propagation* takes a graph in the state after forward propagation finished, and calculates gradients starting from the output towards the input; this direction from the head of the computational graph towards the tail is the result of the calculus chain rule.

## Computational Graph
[ComputationalGraph](../master/src/core/computational_graph.py) class is equipped with methods representing primitive algeabric operations. Each method takes an input and produces an output. Inputs and outputs are represented by the [Connection](../master/src/core/nodes.py) class, and operations by the [Node](../master/scr/core/nodes.py) class. There are two types of connections: constants and variables. The former do not change during the model optimization, but the latter could be changed during the optimization process. Here is the example of the primitive computational graph which is adding two numbers:

```python
from computational_graph import *

cg = ComputationalGraph()
sum = cg.sum(cg.constant(1), cg.constant(2))
```
The code listed above build the computational graph, but doesn't execute it. In order to execute the graph the [SimulationContext](../master/src/core/simulation.py) class should be used. The simulation context has the logic for doing forward/backward propagation. In addition, it stores all computation results produced by each and every operation, including gradients obtained during the backward phase. The code executing the computational graph described above:

```python
from computational_graph import *
from simulation import *

cg = ComputationalGraph()
sum = cg.sum(cg.constant(1), cg.constant(2))

ctx = SimulationContext()
ctx.forward(cg)

print("1+2={}".format(ctx[sum].value))
```

## Loss Functions
Loss functions are used as a mesure of the model performance. Usually, it is just a scalar value telling how well a model estimates output based on the input. Needless to say, a universal loss function which fits all model flavours doesn't exists. The following loss functions are implemented in the [losses](../master/src/core/losses.py) module:

* [Softmax](https://en.wikipedia.org/wiki/Softmax_function)
* [Hinge](https://en.wikipedia.org/wiki/Hinge_loss)

## Usage Examples

> The set of primitive building blocks provided by the framework could be used to build robust estimators. The benefit of using the framework is that you do not have to implement forward/backward propagation from scratch for every kind of an estimator.

* Support Vector Machine
  * [Iris dataset](../master/src/core/examples/iris_svm_classification.ipynb)
* Multilayer Perceptron
  * [Iris dataset](../master/src/core/examples/iris_2_layer_neural_network_classification.ipynb)

from sklearn import datasets
from pydeeptoy.asserts import *
from pydeeptoy.networks import *
from pydeeptoy.optimizers import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
import unittest
from pydeeptoy.losses import *


class SgdOptimizerTest(unittest.TestCase):
    @staticmethod
    def load_mnist_data():
        # load MNIST sample data
        mnist = datasets.load_digits()
        X = np.array(mnist.data)
        y = np.array(mnist.target)
        one_hot_y = np.zeros(shape=(len(np.unique(y)), X.shape[0]))
        for i in range(0, X.shape[0]):
            one_hot_y[y[i], i] = 1
        mean, std = np.mean(X), np.std(X)
        X = (X - mean) / std
        return (X, y, one_hot_y)

    def test_overfit_mnist_with_neural_network(self):
        np.random.seed(100)
        (X, y, one_hot_y) = self.load_mnist_data()

        cg = ComputationalGraph()
        x_in = cg.constant(name="X.T")
        nn_output = neural_network(cg, x_in, X.shape[1], 64, 10)
        nn_output.name = "nn_output"

        y_train = cg.constant(name="one_hot_y")
        loss = softmax(cg, nn_output, y_train, "loss_softmax")

        ctx = SimulationContext()

        sgd = MomentumSgdOptimizer(learning_rate=0.01)
        batch_size=512
        for epoch in range(0, 50):
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            train_x = X[indexes]
            train_y = one_hot_y[:, indexes]
            for batch in range(0, len(train_x), batch_size):
                batch_x = train_x[batch:batch + batch_size]
                batch_y = train_y[:, batch:batch + batch_size]
                sgd.minimize(ctx, cg, {x_in: batch_x.T, y_train: batch_y})

        ctx.forward(cg, {x_in: X.T}, out=[nn_output])
        y_pred = np.argmax(ctx[nn_output].value, axis=0)
        accuracy = np.sum(y_pred == y) / len(y)

        self.assertGreater(accuracy, .7)

    def test_overfit_iris_with_neural_network(self):
        np.random.seed(100)
        iris = load_iris()
        X = iris.data
        y = iris.target

        # standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std
        one_hot_y = np.array(LabelBinarizer().fit_transform(y).T)

        cg = ComputationalGraph()
        x_in = cg.constant(name="X.T")
        nn_output = neural_network(cg, x_in, X.shape[1], 64, 3)
        nn_output.name = "nn_output"

        y_train = cg.constant(name="one_hot_y")
        loss = softmax(cg, nn_output, y_train, "loss_softmax")

        ctx = SimulationContext()
        sgd = MomentumSgdOptimizer(learning_rate=0.05)
        batch_size=256
        for epoch in range(0, 50):
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            train_x = X[indexes]
            train_y = one_hot_y[:, indexes]
            for batch in range(0, len(train_x), batch_size):
                batch_x = train_x[batch:batch + batch_size]
                batch_y = train_y[:, batch:batch + batch_size]
                sgd.minimize(ctx, cg, {x_in: batch_x.T, y_train: batch_y})

        ctx.forward(cg, {x_in: X.T}, out=[nn_output])
        y_pred = np.argmax(ctx[nn_output].value, axis=0)
        accuracy = np.sum(y_pred == y) / len(y)

        accuracy = np.sum(y_pred == y) / len(y)
        self.assertGreater(accuracy, 0.7)

    def test_overfit_iris_with_svm(self):
        np.random.seed(100)
        iris = load_iris()
        X = iris.data
        y = iris.target

        # standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std
        one_hot_y = np.array(LabelBinarizer().fit_transform(y).T)

        cg = ComputationalGraph()
        x_in = cg.constant(name="X.T")
        W = cg.variable("W", 0.01 * np.random.randn(3, X.shape[1]))
        svm_output = cg.matrix_multiply(W, x_in)
        svm_output.name = "svm_output"

        y_train = cg.constant(name="one_hot_y")
        loss = hinge(cg, svm_output, y_train, "loss_hinge")

        ctx = SimulationContext()
        sgd = SgdOptimizer(learning_rate=0.01)
        batch_size=256
        for epoch in range(0, 50):
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            train_x = X[indexes]
            train_y = one_hot_y[:, indexes]
            for batch in range(0, len(train_x), batch_size):
                batch_x = train_x[batch:batch + batch_size]
                batch_y = train_y[:, batch:batch + batch_size]
                sgd.minimize(ctx, cg, {x_in: batch_x.T, y_train: batch_y})

        ctx.forward(cg, {x_in: X.T}, out=[svm_output])
        y_pred = np.argmax(ctx[svm_output].value, axis=0)
        accuracy = np.sum(y_pred == y) / len(y)

        self.assertGreater(accuracy, 0.79)
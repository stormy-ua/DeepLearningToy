from sklearn import datasets
from pydeeptoy.asserts import *
from pydeeptoy.networks import *
from pydeeptoy.optimizers import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
import unittest
from pydeeptoy.losses import *
from sklearn.metrics import accuracy_score


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
        x_in = cg.constant(name="X")
        nn_output = neural_network(cg, x_in, X.shape[1], 64, 30, 10)
        nn_output.name = "nn_output"

        y_train = cg.constant(name="one_hot_y")
        batch_size = 256
        loss = softmax(cg, nn_output, y_train, batch_size, "loss_softmax")

        ctx = SimulationContext()

        sgd = MomentumSgdOptimizer(learning_rate=0.1)
        epochs = 40
        for epoch in range(0, epochs):
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            train_x = X[indexes]
            train_y = one_hot_y[:, indexes]
            for batch in range(0, len(train_x), batch_size):
                batch_x = train_x[batch:batch + batch_size]
                batch_y = train_y[:, batch:batch + batch_size]
                sgd.minimize(ctx, cg, {x_in: batch_x, y_train: batch_y})

        ctx.forward(cg, {x_in: X, y_train: 1})
        y_pred = np.argmax(ctx[nn_output].value, axis=0)
        accuracy = accuracy_score(y, y_pred)

        self.assertEqual(accuracy, 1.)

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
        x_in = cg.constant(name="X")
        nn_output = neural_network(cg, x_in, X.shape[1], 64, 20, 3)
        nn_output.name = "nn_output"

        y_train = cg.constant(name="one_hot_y")
        batch_size = 256
        loss = softmax(cg, nn_output, y_train, batch_size, "loss_softmax")

        ctx = SimulationContext()
        sgd = MomentumSgdOptimizer(learning_rate=0.05)
        batch_size=256
        for epoch in range(0, 500):
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            train_x = X[indexes]
            train_y = one_hot_y[:, indexes]
            for batch in range(0, len(train_x), batch_size):
                batch_x = train_x[batch:batch + batch_size]
                batch_y = train_y[:, batch:batch + batch_size]
                sgd.minimize(ctx, cg, {x_in: batch_x, y_train: batch_y})

        ctx.forward(cg, {x_in: X, y_train: 1})
        y_pred = np.argmax(ctx[nn_output].value, axis=0)
        accuracy = accuracy_score(y, y_pred)

        self.assertGreater(accuracy, 0.95)

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
        x_in = cg.constant(name="X")
        W = cg.variable("W", 0.01 * np.random.randn(3, X.shape[1]))
        svm_output = cg.matrix_multiply(W, cg.transpose(x_in, 1, 0))
        svm_output.name = "svm_output"

        y_train = cg.constant(name="one_hot_y")
        batch_size = 256
        loss = hinge(cg, svm_output, y_train, batch_size, "loss_hinge")

        ctx = SimulationContext()
        sgd = SgdOptimizer(learning_rate=0.05)
        for epoch in range(0, 500):
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            train_x = X[indexes]
            train_y = one_hot_y[:, indexes]
            for batch in range(0, len(train_x), batch_size):
                batch_x = train_x[batch:batch + batch_size]
                batch_y = train_y[:, batch:batch + batch_size]
                sgd.minimize(ctx, cg, {x_in: batch_x, y_train: batch_y})

        ctx.forward(cg, {x_in: X, y_train: 1})
        y_pred = np.argmax(ctx[svm_output].value, axis=0)
        accuracy = accuracy_score(y, y_pred)

        self.assertGreater(accuracy, 0.8)

    # def test_tmp(self):
    #     np.random.seed(100)
    #     (X, y, one_hot_y) = self.load_mnist_data()
    #
    #     cg = ComputationalGraph()
    #     x_in = cg.constant(name="X.T")
    #     # nn_output = neural_network(cg, x_in, X.shape[1], 64, 30, 10)
    #     # nn_output.name = "nn_output"
    #
    #     batch_size = 256
    #
    #     y_train = cg.constant(name="one_hot_y")
    #     x_in_conv1 = cg.reshape(x_in, (batch_size, 1, 8, 8), name="x_in_conv1")
    #     conv1_d = 1
    #     conv1_f = 2
    #     conv1_fn = 3
    #     conv_filter1 = cg.variable(init_value=np.random.randn(conv1_f * conv1_f * conv1_d, conv1_fn))
    #     conv1 = cg.conv2d(x_in_conv1, conv_filter1, receptive_field_size=2, stride=2, padding=0,
    #                       filters_number=conv1_fn)  # (n, d, 3, 3)
    #     conv1_output = cg.reshape(conv1, (batch_size, -1), name="conv1_output")
    #     nn_output = neural_network(cg, cg.transpose(conv1_output, 1, 0), conv1_fn * 4 * 4, 10)
    #     loss = softmax(cg, nn_output, y_train, batch_size, "loss_softmax")
    #
    #     ctx = SimulationContext()
    #
    #     sgd = MomentumSgdOptimizer(learning_rate=0.01)
    #     epochs = 1000
    #     #bar = pyprind.ProgBar(epochs, bar_char='█', width=60, track_time=True, stream=1)
    #     for epoch in range(0, epochs):
    #         indexes = np.arange(0, len(X))
    #         np.random.shuffle(indexes)
    #         train_x = X[indexes]
    #         train_y = one_hot_y[:, indexes]
    #         for batch in range(0, len(train_x), batch_size):
    #             batch_x = train_x[batch:batch + batch_size]
    #             batch_y = train_y[:, batch:batch + batch_size]
    #             sgd.minimize(ctx, cg, {x_in: batch_x.T, y_train: batch_y})
    #         #bar.update(item_id="loss = {0:.5f}".format(ctx[loss].value))
    #
    #     ctx.forward(cg, {x_in: X.T, y_train: 1})
    #     y_pred = np.argmax(ctx[nn_output].value, axis=0)
    #     accuracy = accuracy_score(y, y_pred)
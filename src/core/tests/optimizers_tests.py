from sklearn import datasets
from asserts import *
from networks import *
from optimizers import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer


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

    def test_overfit(self):
        (X, y, one_hot_y) = self.load_mnist_data()
        network = NeuralNetwork(X.shape[1], 64, 10)

        ctx = SimulationContext()

        for w in network.weights:
            ctx[w] = ConnectionData(0.01 * np.random.randn(*w.shape))

        for b in network.biases:
            ctx[b] = ConnectionData(0.01 * np.ones(b.shape))

        sgd = SgdOptimizer(network.network_cg, network.cost_cg)
        for epoch in range(0, 500):
            sgd.minimize(ctx, network.x_in, network.one_hot_y_in, X, one_hot_y, learning_rate=0.05)

        y_pred = np.argmax(network.predict(ctx, X), axis=0)
        accuracy = np.sum(y_pred == y) / len(y)

        self.assertAlmostEqual(accuracy, 1)

    def test_overfit2(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        # standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std
        one_hot_y = np.array(LabelBinarizer().fit_transform(y).T)

        network = NeuralNetwork(X.shape[1], 64, 3)

        for epoch in range(0, 300):
            sgd(network, X, one_hot_y, learning_rate=0.01)

        network.one_hot_y_in.value = one_hot_y
        y_pred = np.argmax(network.predict(X), axis=0)

        accuracy = np.sum(y_pred == y) / len(y)
        self.assertGreater(accuracy, 0.79)

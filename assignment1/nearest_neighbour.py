import numpy as np
from matplotlib import pyplot as plt


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


class Sample:
    def __init__(self, x_arr, label):
        self.x_arr = x_arr
        self.label = label


class KNNClassifier:
    def __init__(self, k: int, x_train: np.array, y_train: np.array):
        self.k = k
        self.samples = [Sample(x_train[i], y_train[i]) for i in range(len(x_train))]

    def predict(self, x):
        k_nearest = self.samples.copy()
        k_nearest.sort(key=lambda sample: np.linalg.norm(sample.x_arr - x))
        labels_k_nearest = [int(sample.label) for sample in k_nearest][:self.k]
        return np.argmax(np.bincount(np.array(labels_k_nearest)))


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return KNNClassifier(k, x_train, y_train)


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    n = x_test.shape[0]
    y_pred = [classifier.predict(x) for x in x_test]
    return np.array(y_pred).reshape(n, 1)


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def error_test_sample_size():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    x_list = [train2, train3, train5, train6]
    y_list = [2, 3, 5, 6]

    x_list_test = [test2, test3, test5, test6]
    len_x_test = len(test2) + len(test3) + len(test5) + len(test6)
    assert len_x_test == np.vstack(x_list_test).shape[0]
    x_test, y_test = gensmallm(x_list_test, y_list, len_x_test)

    sample_sizes = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    errors_avg = []
    errors_min = []
    errors_max = []
    for sample_size in sample_sizes:
        errors = []
        for _ in range(10):
            x_train, y_train = gensmallm(x_list, y_list, sample_size)
            classifier_nn = learnknn(1, x_train, y_train)
            y_test_predict = predictknn(classifier_nn, x_test)
            error = np.mean(np.vstack(y_test) != np.vstack(y_test_predict))
            errors.append(error)
        error_avg = sum(errors) / len(errors)
        errors_avg.append(error_avg)
        errors_min.append(min(errors))
        errors_max.append(max(errors))

    sample_sizes = np.array(sample_sizes)
    errors_avg = np.array(errors_avg)
    errors_min = np.array(errors_min)
    errors_max = np.array(errors_max)
    y_err = [errors_avg - errors_min, errors_max - errors_avg]

    plt.errorbar(sample_sizes, errors_avg, yerr=y_err, label='average error', fmt='-o')
    plt.xlabel('sample size')
    plt.ylabel('test error')
    plt.title('Test Error as a function of Sample Size')
    plt.legend()
    plt.show()


def error_test_k():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    x_list = [train2, train3, train5, train6]
    y_list = [2, 3, 5, 6]

    x_list_test = [test2, test3, test5, test6]
    len_x_test = len(test2) + len(test3) + len(test5) + len(test6)

    x_test, y_test = gensmallm(x_list_test, y_list, len_x_test)

    sample_size = 200
    ks = [i for i in range(1, 12)]
    errors_avg = []
    errors_min = []
    errors_max = []
    for k in ks:
        errors = []
        for _ in range(10):
            x_train, y_train = gensmallm(x_list, y_list, sample_size)
            classifier_knn = learnknn(k, x_train, y_train)
            y_test_predict = predictknn(classifier_knn, x_test)
            error = np.mean(np.vstack(y_test) != np.vstack(y_test_predict))
            errors.append(error)
        error_avg = sum(errors) / len(errors)
        errors_avg.append(error_avg)
        errors_min.append(min(errors))
        errors_max.append(max(errors))

    ks = np.array(ks)
    errors_avg = np.array(errors_avg)
    errors_min = np.array(errors_min)
    errors_max = np.array(errors_max)
    y_err = [errors_avg - errors_min, errors_max - errors_avg]

    plt.errorbar(ks, errors_avg, yerr=y_err, label='average error', fmt='-o')
    plt.xlabel('k')
    plt.ylabel('test error')
    plt.title('Test Error as a function of K in k-nearest-neighbors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    error_test_sample_size()
    error_test_k()

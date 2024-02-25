import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m = trainX.shape[0]
    d = trainX.shape[1]
    H0 = np.eye(d)
    H1 = np.zeros((d, m))
    H2 = np.zeros((m, d))
    H3 = np.zeros((m, m))
    H = matrix(np.block([[H0, H1], [H2, H3]]) * (2 * l))
    u0 = np.zeros((d, 1))
    u1 = np.ones((m, 1)) * (1 / m)
    u = matrix(np.block([[u0], [u1]]))
    v0 = np.ones((m, 1))
    v1 = np.zeros((m, 1))
    v = matrix(np.block([[v0], [v1]]))
    Ymat = np.diag(trainy)
    Xmat = trainX
    A0 = np.dot(Ymat, Xmat)
    A1 = np.eye(m)
    A2 = np.zeros((m, d))
    A3 = np.eye(m)
    A = matrix(np.block([[A0, A1], [A2, A3]]))
    sol = solvers.qp(H, u, -A, -v)
    return np.array(sol["x"][:d])


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def softsvm_predict(w, x):
    """

    :param w: a row vector
    :param x:  a line vector
    :return:
    """
    return np.sign(x @ w)


def softsvm_predict_test(w, x_test: np.array):
    n = x_test.shape[0]
    y_pred = [softsvm_predict(w, x) for x in x_test]
    return np.array(y_pred).reshape(n, 1)


def plot_error(sample_size, n_array, experiments):
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = sample_size
    d = trainX.shape[1]

    errors_train_avg = []
    errors_train_min = []
    errors_train_max = []
    errors_test_avg = []
    errors_test_min = []
    errors_test_max = []
    for n in n_array:
        l = 10 ** n
        errors_train = []
        errors_test = []
        for _ in range(experiments):
            # Get a random m training examples from the training set
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:m]]
            _trainy = trainy[indices[:m]]

            # run the softsvm algorithm
            w = softsvm(l, _trainX, _trainy)

            y_sample_predict = softsvm_predict_test(w, _trainX)
            y_test_predict = softsvm_predict_test(w, testX)

            error_train = np.mean(np.vstack(_trainy) != np.vstack(y_sample_predict))
            error_test = np.mean(np.vstack(testy) != np.vstack(y_test_predict))
            errors_train.append(error_train)
            errors_test.append(error_test)

        error_train_avg = sum(errors_train) / len(errors_train)
        errors_train_avg.append(error_train_avg)
        errors_train_min.append(min(errors_train))
        errors_train_max.append(max(errors_train))

        error_test_avg = sum(errors_test) / len(errors_test)
        errors_test_avg.append(error_test_avg)
        errors_test_min.append(min(errors_test))
        errors_test_max.append(max(errors_test))

    n_array = np.array(n_array)
    errors_train_avg = np.array(errors_train_avg)
    errors_train_min = np.array(errors_train_min)
    errors_train_max = np.array(errors_train_max)
    errors_test_avg = np.array(errors_test_avg)
    errors_test_min = np.array(errors_test_min)
    errors_test_max = np.array(errors_test_max)

    y_err_train = [errors_train_avg - errors_train_min, errors_train_max - errors_train_avg]
    y_err_test = [errors_test_avg - errors_test_min, errors_test_max - errors_test_avg]

    fmt = '-o'
    label_train = 'train average error'
    label_test = 'test average error'
    if len(n_array) < 5:
        fmt = 'o'
        label_train = 'train error'
        label_test = 'test error'
        y_err_train = None
        y_err_test = None

    plt.errorbar(n_array, errors_train_avg, yerr=y_err_train, label=label_train, fmt=fmt, elinewidth=3,
                 linewidth=3)
    plt.errorbar(n_array, errors_test_avg, yerr=y_err_test, label=label_test, fmt=fmt, elinewidth=2,
                 linewidth=2)

    plt.xlabel('n')
    plt.ylabel('error')
    plt.title('Error as a function of l, while l = 10^n')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    simple_test()

    plot_error(100, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10)

    plot_error(1000, [1, 3, 5, 8], 1)

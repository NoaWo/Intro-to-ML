import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


def inner_product(x, y):
    return np.dot(x, y)


def poly_kernel(x1, x2, k):
    return (1 + inner_product(x1, x2)) ** k


def compute_gram_matrix(X, K):
    m = X.shape[0]
    gram_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            gram_matrix[i, j] = K(X[i], X[j])

    return gram_matrix


def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    trainX = trainX.astype(np.float64)
    trainy = trainy.astype(np.float64)

    m = trainX.shape[0]
    d = trainX.shape[1]
    K = lambda x1, x2: poly_kernel(x1, x2, k)

    I_2m = np.eye(2 * m)
    epsilon = 0.01

    G = compute_gram_matrix(trainX, K)

    H1 = np.zeros((m, m))
    H2 = np.zeros((m, m))
    H3 = np.zeros((m, m))
    H = matrix((np.block([[G, H1], [H2, H3]]) * (2 * l)) + (I_2m * epsilon))

    u0 = np.zeros((m, 1))
    u1 = np.ones((m, 1)) * (1 / m)
    u = matrix(np.block([[u0], [u1]]))

    v0 = np.ones((m, 1))
    v1 = np.zeros((m, 1))
    v = matrix(np.block([[v0], [v1]]))

    Ymat = np.diag(trainy)
    A0 = np.dot(Ymat, G)
    A1 = np.eye(m)
    A2 = np.zeros((m, m))
    A3 = np.eye(m)
    A = matrix(np.block([[A0, A1], [A2, A3]]))

    sol = solvers.qp(H, u, -A, -v)
    return np.array(sol["x"][:m])


def alphas_predict(x, alpha_vec, K, X):
    sum = 0

    for j, alpha_j in enumerate(alpha_vec):
        sum += alpha_j * K(X[j], x)

    return np.sign(sum).item()


def report_results(results):
    print("Results:")
    for result in results:
        print(f"for l={result[1]},k={result[2]} average validation error is {result[0]}")


def n_fold_cross_validation_softsvm(trainX, trainy, ls, ks, n=5):
    results = []

    part_len = len(trainX) // n
    extra_xs = len(trainX) % n
    part_len_list = [part_len + 1 if i < extra_xs else part_len for i in range(n)]
    start_of_part_list = [sum(part_len_list[:i]) for i in range(n)]

    # split into parts
    x_parts = [trainX[start_of_part_list[i]:start_of_part_list[i] + part_len_list[i]] for i in range(n)]
    y_parts = [trainy[start_of_part_list[i]:start_of_part_list[i] + part_len_list[i]] for i in range(n)]

    # find errors for each l,k pair:
    for l in ls:
        for k in ks:
            K = lambda x1, x2: poly_kernel(x1, x2, k)
            errors = []
            for i in range(n):
                # v
                x_v = x_parts[i]
                y_v = y_parts[i]

                # s'
                x_parts_without_i = [x_part for index, x_part in enumerate(x_parts) if index != i]
                x_s_tag = np.concatenate(x_parts_without_i, axis=0)
                y_parts_without_i = [element for index, element in enumerate(y_parts) if index != i]
                y_s_tag = np.concatenate(y_parts_without_i, axis=0)

                # run softsvm
                predictor = softsvmpoly(l, k, x_s_tag, y_s_tag)

                # calculate errors
                err = 0
                for xi, yi in zip(x_v, y_v):
                    if alphas_predict(xi, predictor, K, x_s_tag) != yi:
                        err += 1
                err = err / len(x_v)
                errors.append(err)
            err_res = sum(errors) / len(errors)
            results.append((err_res, l, k))

    errors_by_l_k = [result[0] for result in results]
    best_index = errors_by_l_k.index(min(errors_by_l_k))
    best_l = results[best_index][1]
    best_k = results[best_index][2]

    best_pred = softsvmpoly(best_l, best_k, trainX, trainy)
    report_results(results)
    print(f"Best predictor is for l={best_l}, k={best_k}")
    return best_pred, best_l, best_k


def q2a(trainX, trainy):
    """
    plot the points in the training set in R2 and color them by their label.
    """
    cdict = {-1: 'red', 1: 'blue'}
    ctrainy = [cdict[yi] for yi in trainy]

    plt.scatter([vec[0] for vec in trainX], [vec[1] for vec in trainX], c=ctrainy)
    plt.xlabel('x(1)')
    plt.ylabel('x(2)')
    plt.title('Sample')
    # add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=cdict[label], markersize=10) for
        label in [-1, 1]]
    plt.legend(handles=handles)
    plt.grid(True)
    plt.show()


def q2b(trainX, trainy, testX, testy):
    best_pred, best_l, best_k = n_fold_cross_validation_softsvm(trainX, trainy, [1, 10, 100], [2, 5, 8])
    K = lambda x1, x2: poly_kernel(x1, x2, best_k)

    err = 0
    size = len(testX)

    for xi, yi in zip(testX, testy):
        if alphas_predict(xi, best_pred, K, trainX) != yi:
            err += 1

    test_error = err / size
    print(f'Test error of the resulting classifier is {test_error}')


def q2d(trainX, trainy, testX, testy):
    l = 100
    ks = [3, 5, 8]

    # define a fixed region for visualization
    x_min, x_max = np.min(trainX[:, 0]) - 0.3, np.max(trainX[:, 0]) + 0.3
    y_min, y_max = np.min(trainX[:, 1]) - 0.3, np.max(trainX[:, 1]) + 0.3

    # generate a fine grid for visualization
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    cdict = {-1: 'red', 1: 'blue'}
    ctrainy = [cdict[yi] for yi in trainy]
    ctesty = [cdict[yi] for yi in testy]

    for k in ks:
        K = lambda x1, x2: poly_kernel(x1, x2, k)
        predictor = softsvmpoly(l, k, trainX, trainy)

        # predict labels for each point in the meshgrid
        Z = np.zeros_like(xx, dtype=int)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = alphas_predict(np.array([xx[i, j], yy[i, j]]), predictor, K, trainX)

        # plot the decision regions
        plt.imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap='RdYlBu', aspect='auto',
                   origin='lower', alpha=0.5)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        # plot the training points
        plt.scatter([vec[0] for vec in trainX], [vec[1] for vec in trainX], c=ctrainy)
        # plot the test points
        plt.scatter([vec[0] for vec in testX], [vec[1] for vec in testX], c=ctesty)

        plt.xlabel('x(1)')
        plt.ylabel('x(2)')

        # add legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=cdict[label], markersize=10) for
            label in [-1, 1]]
        plt.legend(handles=handles)

        plt.title(f'Degree {k} Polynomial Soft SVM')
        plt.show()


def simple_test():
    # load question 2 data
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    simple_test()

    # load question 2 data
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    q2a(trainX, trainy)
    q2b(trainX, trainy, testX, testy)
    q2d(trainX, trainy, testX, testy)

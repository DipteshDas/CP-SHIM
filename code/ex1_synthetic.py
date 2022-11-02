from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import stats_all

import compute_CP

import sys


def read_sample(sample_path):
    npzfile = np.load(sample_path, allow_pickle=True)
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']

    X_test = npzfile['X_test']
    y_test = npzfile['y_test']

    return X_train, y_train, X_test, y_test


def cp_split(X_tr, y_tr, X_te, model, sig_level):
    X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, test_size=0.5, random_state=0)

    # Train with half the data ('X_train', 'y_train')
    model.fit(X_train, y_train)

    # Inference with remaining half data ('X_valid', 'y_valid')
    y_valid_pred = model.predict(X_valid)
    res = np.abs(y_valid - y_valid_pred)

    # Prediction for the 'test data'
    y_te_pred = model.predict(X_te)

    # Ranking on the calibration set
    sorted_residual = np.sort(res)
    index = int((X_tr.shape[0] / 2 + 1) * (1 - sig_level))

    # pdb.set_trace()
    L = sorted_residual[index]

    return y_te_pred, L


def comp_cp_set(y_te_pred, L):
    lb = y_te_pred - L
    ub = y_te_pred + L
    cp_set = np.array([lb.tolist(), ub.tolist()]).T.tolist()

    return cp_set


def opt_lmd_shim(dir_ms, max_depth):
    lmd_path_opt = dir_ms + 'opt_lmd_ord_' + str(max_depth) + '.npz'  # filepath to 'optimum lmd'.
    npzfile = np.load(lmd_path_opt, allow_pickle=True)
    lmd = npzfile['lmd_opt']

    return lmd


def stat_shim(X_tr, y_tr, X_te, y_te, split=False, lmd=1.0, alpha=0.001, max_depth=2):
    sig_level = 0.1
    n = X_te.shape[0]
    count = 0
    L = []
    y_preds = []
    for i in range(n):
        if split:
            cp_set, y_pred_homo = compute_CP.cp_split(X_tr, y_tr, X_te[i][:, np.newaxis], lmd, alpha,
                                                      max_depth, sig_level)
        else:
            cp_set, y_pred_homo, _, _ = compute_CP.cp_homotopy(X_tr, y_tr, X_te[i][:, np.newaxis], lmd, alpha,
                                                               max_depth, sig_level)
        # pdb.set_trace()
        if cp_set[0] <= y_te[i] <= cp_set[1]:
            count += 1

        cl_homo = cp_set[1] - cp_set[0]

        L += [cl_homo]
        y_preds += [y_pred_homo]

        # if i % 10 == 0:
        #     print("{} test examples done...\n".format(i))

    cov = count / n
    r2 = r2_score(y_te, y_preds)

    return cov, L, r2


def stat(X_train, y_train, X_test, y_test, model):
    n = y_test.shape[0]

    # Compute the 'split CP'
    y_te_pred, L = cp_split(X_train, y_train, X_test, model, sig_level=0.1)
    cp_set = comp_cp_set(y_te_pred, L)

    # Compute the 'r2score'
    r2 = r2_score(y_test, y_te_pred)

    count = 0
    for i in range(n):
        if cp_set[i][0] <= y_test[i] <= cp_set[i][1]:
            count += 1

    cov = count / n

    return cov, 2 * L, r2


def data(name, i):
    [X_train, y_train, X_test, y_test] = None

    if name == "synthetic1":
        sample_path = "../data/synthetic/low/zeta_0.4/cv2/sample_" + str(i) + "/" + "data.npz"
        X_train, y_train, X_test, y_test = read_sample(sample_path)
    if name == "compas":
        sample_path = "../data/real/compas/cv/sample_" + str(i) + "/" + "data.npz"
        X_train, y_train, X_test, y_test = read_sample(sample_path)

    return X_train, y_train, X_test, y_test


def load_opt_param_mlp(param_path):
    npzfile = np.load(param_path, allow_pickle=True)
    hl = npzfile["hl"].tolist()
    act = npzfile["act"].tolist()

    return hl, act


def load_opt_param_rf(param_path):
    npzfile = np.load(param_path, allow_pickle=True)
    msl = npzfile["msl"].item()
    n_est = npzfile["n_est"].item()

    return msl, n_est


def call_method(dir_ms, dir_cv, n_sample=15, name="rf", split=False, ord=1):
    lamda = 1.0
    sf = 'f'
    model = None

    if name == "shim":
        max_depth = ord
        lamda = opt_lmd_shim(dir_ms, max_depth=max_depth)
        if split:
            sf = 's'
        print("shim_{}{}...\n".format(ord, sf))

    if name == "mlp":
        print("mlp...\n")
        param_path = dir_ms + "opt_param_mlp.npz"
        hl, act = load_opt_param_mlp(param_path)

        model = MLPRegressor(hidden_layer_sizes=hl, activation=act, random_state=0, max_iter=2000)

    elif name == "rf":
        print("rf...\n")
        param_path = dir_ms + "opt_param_rf.npz"
        msl, n_est = load_opt_param_rf(param_path)
        model = RandomForestRegressor(n_jobs=-1, n_estimators=n_est, min_samples_leaf=msl, warm_start=False,
                                      random_state=0)

    covs, cpls, r2s = [], [], []
    for i in range(n_sample):
        sample_path = dir_cv + "sample_" + str(i) + "/" + "data.npz"
        X_train, y_train, X_test, y_test = read_sample(sample_path)

        if name == "shim":
            # cov, cpl, r2 = stat_shim(X_train, y_train, X_test, y_test, split, lmd=lamda, alpha=0.001, max_depth=ord)
            cov, cpl, r2 = stats_all.stat_shim(X_train, y_train, X_test, y_test, split, lmd=lamda, alpha=0.001,
                                               max_depth=ord)
        else:
            # cov, cpl, r2 = stat(X_train, y_train, X_test, y_test, model)
            cov, cpl, r2 = stats_all.stat(X_train, y_train, X_test, y_test, model)

        print("sample: {}, cov: {}, cpl: {}\n".format(i, np.mean(cov), np.mean(cpl)))
        sys.stdout.flush()

        covs += cov
        cpls += cpl
        r2s += r2

    print("cov: {}({}), cpl: {}({}), r2: {}({})\n\n".format(np.mean(covs), np.std(covs), np.mean(cpls), np.std(cpls),
                                                            np.mean(r2s), np.std(r2s)))

    print("\n\n====================================================================\n\n")

    sys.stdout.flush()


def choose_data(name='vl3', zeta=0.04):
    dir_ms = "../data/synthetic/" + str(name) + "/" + "zeta_" + str(zeta) + "/ms/"
    dir_cv = "../data/synthetic/" + str(name) + "/" + "zeta_" + str(zeta) + "/cv/"

    return dir_ms, dir_cv


def run(name='low'):
    dir_ms, dir_cv = choose_data(name=name, zeta=0.4)

    call_method(dir_ms, dir_cv, n_sample=15, name="mlp")
    call_method(dir_ms, dir_cv, n_sample=15, name="rf")

    call_method(dir_ms, dir_cv, n_sample=15, name="shim", split=True, ord=1)
    call_method(dir_ms, dir_cv, n_sample=15, name="shim", split=False, ord=1)

    call_method(dir_ms, dir_cv, n_sample=15, name="shim", split=True, ord=2)
    call_method(dir_ms, dir_cv, n_sample=15, name="shim", split=False, ord=2)

    call_method(dir_ms, dir_cv, n_sample=15, name="shim", split=True, ord=3)
    call_method(dir_ms, dir_cv, n_sample=15, name="shim", split=False, ord=3)


if __name__ == "__main__":
    run('low')  # low dimensional data
    run('high')  # high dimensional data

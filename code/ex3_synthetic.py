import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import stats_all
import pdb


def plot(data_mlp, data_rf, data_lf, data_ls, data_sf, data_ss, n_samples, xlabel, ylabel, fname, ylim):
    no_labels = [''] * len(n_samples)
    # pdb.set_trace()
    bp_mlp = plt.boxplot(data_mlp, patch_artist=True, positions=[3, 12, 21], labels=no_labels, boxprops=dict(facecolor="C0", alpha=0.5))

    bp_rf = plt.boxplot(data_rf, patch_artist=True, positions=[4, 13, 22], labels=no_labels,
                        boxprops=dict(facecolor="C1", alpha=0.5))

    bp_ls = plt.boxplot(data_ls, patch_artist=True, positions=[5, 14, 23], labels=n_samples,
                        boxprops=dict(facecolor="C2", alpha=0.5))

    bp_lf = plt.boxplot(data_lf, patch_artist=True, positions=[6, 15, 24], labels=no_labels,
                        boxprops=dict(facecolor="C3", alpha=0.5))

    bp_ss = plt.boxplot(data_ss, patch_artist=True, positions=[7, 16, 25], labels=no_labels,
                        boxprops=dict(facecolor="C4", alpha=0.5))

    bp_sf = plt.boxplot(data_sf, patch_artist=True, positions=[8, 17, 26], labels=no_labels,
                        boxprops=dict(facecolor="C5", alpha=0.5))

    plt.tick_params(bottom=False)

    ax = plt.gca()
    ax.legend([bp_mlp["boxes"][0], bp_rf["boxes"][0], bp_ls["boxes"][0], bp_lf["boxes"][0], bp_ss["boxes"][0],
               bp_sf["boxes"][0]], ['mlp', 'rf', 'lasso_s', 'lasso_f', 'shim_s', 'shim_f'], loc='upper right',
              prop={'size': 16}, ncol=3)

    ax.set_xlim(0, 30)
    ax.set_ylim(ylim)

    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    plt.show()


def read_sample(sample_path):
    npzfile = np.load(sample_path, allow_pickle=True)
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']

    X_test = npzfile['X_test']
    y_test = npzfile['y_test']

    return X_train, y_train, X_test, y_test


def opt_lmd_shim(dir_ms, max_depth):
    lmd_path_opt = dir_ms + 'opt_lmd_ord_' + str(max_depth) + '.npz'  # filepath to 'optimum lmd'.
    npzfile = np.load(lmd_path_opt, allow_pickle=True)
    lmd = npzfile['lmd_opt']

    return lmd


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
    model = None,
    lamda = 1.0

    if name == "shim":
        max_depth = ord
        lamda = opt_lmd_shim(dir_ms, max_depth=max_depth)

        if split:
            print("shim" + str(ord) + "_s...\n")
        else:
            print("shim" + str(ord) + "_f...\n")

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
            cov, cpl, r2 = stats_all.stat_shim(X_train, y_train, X_test, y_test, split, lmd=lamda, alpha=0.001,
                                               max_depth=ord)
        else:
            cov, cpl, r2 = stats_all.stat(X_train, y_train, X_test, y_test, model)

        print("sample: {} completed...\n".format(i))

        cpls += cpl
        covs += cov
        r2s += r2

    return covs, cpls, r2s


def run():
    sample_sizes = [50, 100, 150]
    names = ['ss1', 'ss2', 'ss3']
    zeta = 0.6

    data_mlp, data_rf, data_sf, data_ss, data_lf, data_ls = [], [], [], [], [], []
    r2_mlps, r2_rfs, r2_sfs, r2_sss, r2_lfs, r2_lss = [], [], [], [], [], []

    for i in range(len(sample_sizes)):
        dir_ms = "../data/synthetic/" + str(names[i]) + "/" + "zeta_" + str(zeta) + "/ms/"
        dir_cv = "../data/synthetic/" + str(names[i]) + "/" + "zeta_" + str(zeta) + "/cv/"

        _, cpl_mlp, r2_mlp = call_method(dir_ms, dir_cv, n_sample=3, name="mlp")
        _, cpl_rf, r2_rf = call_method(dir_ms, dir_cv, n_sample=3, name="rf")

        _, cpl_ls, r2_ls = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=True, ord=1)
        _, cpl_lf, r2_lf = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=False, ord=1)

        _, cpl_ss, r2_ss = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=True, ord=5)
        _, cpl_sf, r2_sf = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=False, ord=5)

        data_mlp += [cpl_mlp]
        data_rf += [cpl_rf]

        data_ls += [cpl_ls]
        data_lf += [cpl_lf]

        data_ss += [cpl_ss]
        data_sf += [cpl_sf]

        r2_mlps += [r2_mlp]
        r2_rfs += [r2_rf]

        r2_sss += [r2_ss]
        r2_sfs += [r2_sf]

        r2_lss += [r2_ls]
        r2_lfs += [r2_lf]

        print("====================== sample_size: {} is done ! ================== \n".format(sample_sizes[i]))

    stat_path = "../results/synthetic/stat_ss.npz"
    np.savez(stat_path, data_mlp=data_mlp, data_rf=data_rf, data_lf=data_lf, data_ls=data_ls,
             data_sf=data_sf, data_ss=data_ss, sample_sizes=sample_sizes)

    ylim = (0.5, 3.5)
    xlabel, ylabel, fname = "Sample size", "CI length", "../results/synthetic/figures/cpl_ss.pdf"
    plot(data_mlp, data_rf, data_lf, data_ls, data_sf, data_ss, sample_sizes, xlabel, ylabel, fname, ylim)


if __name__ == "__main__":
    run()

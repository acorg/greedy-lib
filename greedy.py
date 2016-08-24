import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
import warnings
import matplotlib.patches as mpatches
import itertools
from joblib import Parallel, delayed
import multiprocessing as mp
import os
num_cores = mp.cpu_count()
pandas2ri.activate()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r("""
    library(miscTools)
    library(maxLik)
    library(truncreg)
    """)


def coefplot(formula, data, fontsize=5):
    """ Plots coefficients of a regression model.
        formula = patsy-style formula for regression model
        data = pandas dataframe with columns for the variables in the formula
        fontsize = 5 by default
        returns figure, axes
    """
    lm = smf.ols(formula, data=data).fit()
    lm0 = smf.ols(formula + "+ 0", data=data).fit()
    r.assign("data", data)
    r("""
    trunc_reg <- truncreg(%s,
                          data = data,
                          point = 0,
                          direction = 'left')
    summ <- summary(trunc_reg)
    coeffs <- trunc_reg$coefficients
    coeffs <- coeffs[names(coeffs) != "sigma"]
    coeffs_values <- as.vector(coeffs, mode="numeric")
    coeffs_bse <- head(summary(trunc_reg)$coefficients[,2], -1)
    """ % (formula))
    params = pd.DataFrame(data=r("coeffs_values"), index=r("names(coeffs)"))
    params.index = [":".join(sorted(name.split(":"))) for name in params.index]
    params = params.drop("(Intercept)")
    params.columns = ["truncreg"]
    truncreg_bse = pd.DataFrame(data=r("coeffs_bse"), index=r("names(coeffs)"))
    lm_params = lm.params.drop("Intercept")
    lm_params.index = [":".join(sorted(name.split(":"))) for name in
                       lm_params.index]
    params["lm"] = lm_params
    lm0_params = lm0.params
    lm0_params.index = [":".join(sorted(name.split(":"))) for name in
                        lm0_params.index]
    params["lm0"] = lm0_params
    params = params.sort_values("lm")
    lm_bse = lm.bse
    lm_bse.index = [":".join(sorted(name.split(":"))) for name in
                    lm_bse.index]
    lm0_bse = lm0.bse
    lm0_bse.index = [":".join(sorted(name.split(":"))) for name in
                     lm0_bse.index]
    fig, ax = plt.subplots()
    y = range(len(params.index))
    ax.scatter(list(params["truncreg"]), y, color="g", s=2)
    ax.scatter(list(params["lm"]), y, color="r", s=2)
    ax.scatter(list(params["lm0"]), y,  color="b", s=2)
    for y in range(len(params.index)):
        sub = params.index[y]
        x = params.lm[sub]
        se = lm_bse[sub]
        ax.plot([x - se, x + se], [y, y], color="red")
        x = params.lm0[sub]
        se = lm0_bse[sub]
        ax.plot([x - se, x + se], [y, y], color="blue")
        x = params.truncreg[sub]
        for perm in list(itertools.permutations(sub.split(":"))):
            s = ":".join(perm)
            print s
            try:
                se = truncreg_bse.loc[s]
                ax.plot([x - se, x + se], [y, y], color="green")
            except KeyError:
                pass
    red_patch = mpatches.Patch(color='red', label='Linear Model')
    blue_patch = mpatches.Patch(color='blue', label='Forced Zero Intercept')
    green_patch = mpatches.Patch(color='green', label='Truncated Regression')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=2)
    plt.yticks(range(len(params.index)), params.index)
    ax.set_ylim([-1, len(params)])
    ax.set_yticklabels(params.index,
                       fontsize=fontsize)
    ax.set_ylabel("Substitutions")
    ax.set_xlabel("Coefficients")
    plt.title("Coefficient plot")
    plt.grid()
    fig.savefig("coefplot.png", dpi=200)
    file = open("lm_summary.txt", "w")
    file.write(str(lm.summary()))
    file.close()
    file = open("lm0_summary.txt", "w")
    file.write(str(lm0.summary()))
    file.close()
    file = open("truncreg_summary.txt", "w")
    file.write(str(r("summ")))
    file.close()
    return fig, ax


def collinify(data):
    "data = pandas dataframe with individual data.columns in columns"
    "returns new pandas dataframe with culled collinear columns"
    df = pd.DataFrame()
    for predictor in data.columns:
        collinears = []
        for col in data.columns:
            if not np.logical_and(data[predictor] != np.zeros(len(data.index)),
                                  data[col] != data[predictor]).any():
                collinears.append(col)
        df["_".join(sorted(collinears))] = data[predictor]
    return df

candidates = []
predictors = []


def get_bic(candidate, data):
    results = smf.ols(formula="AGDIST ~ %s" %
                      ('+'.join(predictors + [candidate])),
                      data=data).fit()
    if np.isnan(results.tvalues).any():
        candidates.remove(candidate)
    return results.bic


def greedy(singles, data):
    "singles = list of individual candidates for parameters in linear model"
    "data = pandas dataframe with columns AGDIST and singles"
    "returns useful predictors"
    global candidates, predictors
    #  First we check if we need to continue from where we left off.
    if os.path.isfile("predictors/predictors_00.csv"):
        latest = 0
        while True:
            if not os.path.isfile("predictors/predictors_%02d.csv" %
                                  (latest + 1)):
                break
            else:
                latest += 1
        file = open("predictors/predictors_%02d.csv" % (latest), "r")
        predictors = file.read().split("\n")
        file.close()
        file = open("candidates/candidates_%02d.csv" % (latest), "r")
        candidates = file.read().split("\n")
        file.close()
        best_bic = smf.ols(formula="AGDIST ~ %s " % ('+'.join(predictors)),
                           data=data).fit().bic
        loop = latest + 1
    else:
        if not os.path.exists("predictors"):
            os.makedirs("predictors")
        if not os.path.exists("candidates"):
            os.makedirs("candidates")
        pairs = list(itertools.combinations(singles, 2))
        pairs = [x + ":" + y for x, y in pairs]
        candidates = singles + pairs
        best_bic = 0
        loop = 0
    #  Now we run the greedy algorithm using multithreading
    while not os.path.isfile("predictors.csv"):
        print "%d predictors, %d candidates" % (len(predictors),
                                                len(candidates))
        best_candidate = []
        bics = Parallel(n_jobs=num_cores)(delayed(get_bic)(candidate, data)
                                          for candidate in candidates)
        min_bic = min(bics)
        if min_bic < best_bic or best_bic == 0:
            best_bic = min_bic
            best_candidate = candidates[bics.index(min_bic)]
            candidates.remove(best_candidate)
            predictors.append(best_candidate)
            file = open("predictors/predictors_%02d.csv" % (loop), "w")
            file.write("\n".join(predictors))
            file.close()
            file = open("candidates/candidates_%02d.csv" % (loop), "w")
            file.write("\n".join(candidates))
            file.close()
            loop += 1
        else:
            file = open("predictors.csv", "w")
            file.write("\n".join(predictors))
            file.close()
            file = open("candidates.csv", "w")
            file.write("\n".join(candidates))
            file.close()
    #  Save results
    file = open("predictors.csv", "r")
    predictors = file.read().split("\n")
    file.close()
    return predictors


def greedyvis(predictors, data):
    "predictors = list of successively added predictors"
    "data = pandas dataframe with predictors in columns"
    "returns figure, leftaxes, rightaxes"
    bics = []
    coeffs = []
    for i in range(len(predictors)):
        formula = 'AGDIST ~ %s' % ('+'.join(predictors[: (i+1)]))
        results = smf.ols(formula, data=data).fit()
        bics.append(results.bic)
        for comb in itertools.permutations(predictors[i].split(":")):
            try:
                coeffs.append(float(results.params.loc[":".join(comb)]))
            except KeyError:
                pass
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.plot(range(len(bics)), list(bics), color="b")
    plt.xticks(range(len(bics)), predictors, rotation='vertical')
    ax1.tick_params(axis='x', which='major', labelsize=4)
    ax1.tick_params(axis='x', which='minor', labelsize=4)
    ax1.set_xlabel("Amino acid substiution added")
    ax1.set_ylabel("Bayesian Information Criterion", color="b")
    ax1.set_xlim([0, len(bics)])
    plt.title("Change in BIC during forward selection")
    ax2 = ax1.twinx()
    ax2.plot(range(len(coeffs)), coeffs, color="r")
    ax2.set_ylabel("Coefficient in linear regression", color="r")
    fig.savefig("coefvis.png")
    return fig, ax1, ax2


def coefvis(predictor, model):
    """predictor = parameter whose estimate you want to understand
    model = linear model object from statsmodels.formula.api.ols
    saves csvs and heatmap pngs in a folder
    returns figure, axes"""
    results = model.fit()
    print results.params
    coeffs = results.params
    predictors = model.exog_names
    endog = model.endog.reshape((-1, 1))
    exog = model.exog
    data = pd.DataFrame(data=np.hstack((exog, endog)),
                        columns=np.append(model.exog_names,
                                          model.endog_names))
    if "Intercept" in predictors:
        predictors.remove("Intercept")
        data = data.drop("Intercept", axis=1)
    if not os.path.exists(predictor):
        os.makedirs(predictor)
    if not os.path.isfile(predictor + "/" + predictor + ".csv"):
        table = pd.DataFrame()
        for i in range(len(data.index)):
            if data[predictor][i] != 0:
                table = table.append(data.iloc[i, :])
        for j in table:
            if all([x == 0 for x in table[j]]):
                del table[j]
        del table[predictor]
        table.to_csv(predictor + "/" + predictor + ".csv", index=False)
    table = pd.read_csv(predictor + "/" + predictor + ".csv")
    array = table.drop([col for col in data.columns if col not in predictors],
                       axis=1)
    array.index = [round(x, ndigits=2) for x in table[model.endog_names]]
    for a in array.columns:
        array[a] = array[a] * coeffs[a]
    array = array[array.columns].astype(float)
    np_array = np.array(array)
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(np_array)
    plt.colorbar(heatmap)
    length = len(array.columns)
    for y in range(np_array.shape[0]):
        for x in range(np_array.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.1f' % np_array[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=24*2**-(length/8)
                     )
    for y in range(np_array.shape[0]):
        plt.text(-0.5, y + 0.5, '%.1f' % array.index[y],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=24*2**-(length/8))
    for x in range(np_array.shape[1]):
        plt.text(x + 0.5, -1, '%s' % array.columns[x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=24*2**-(length/8))
    plt.tick_params(
                    axis='x',
                    which='both',
                    bottom='on',
                    top='off')
    plt.tick_params(
                    axis='y',
                    which='both',
                    left='on',
                    right='off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("%s=%.1f" % (predictor, coeffs[predictor]))
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(predictor + "/" + predictor + ".png", dpi=500)
    return fig, ax


if __name__ == "__main__":
    data = pd.read_csv("culled.csv", sep=" ")
    data = data.drop("Unnamed: 75", axis=1)
    print collinify(data).columns
    model = smf.ols("AGDIST ~ %s" %
                    ("+".join([x for x in data.columns if x != "AGDIST"])),
                    data=data)
    coefvis("QR197", model)

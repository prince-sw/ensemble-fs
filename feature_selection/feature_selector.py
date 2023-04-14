from feature_selection.supervised import *
from feature_selection.unsupervised import *
from feature_selection.heuristic import *
from observations.model_config import fs_methods, fs_methods_heuristics

feature_selector_method = {
    "corr": corr_selection,
    "tree": treebased_selection,
    "var": variance_selection,
    "lrs": logistic_regression_selection,
    "rfe": rfe_selection,
    "bor": boruta_selection,
    "lass": Lasso_selection,
    "muin": Mutualinfo_selection,
    "chi": chitest_selection,
    "rfc": randomforest_Selection,
    "ulap": laplacian_selection,
    "uilap": iterlaplacian_selection,
    "ucos": cosine_selection,
    "upcorr": pairwise_corr_selection,
    "ufrufs": frufs_selection,
    "em": ensemble_selection,
    "emm": ensemblemd_selection
}

heuristic_feature_selector_method = {
    "gen": genetic_algorithm,
    "ps": particleswarm_algorithm,
    "mrf": mantaray_optimization,
    "gr": goldenratio_optimizer,
    "sm": socialmimic_optimizer
}


def feature_selector(df, label, k):
    features_selected = {}
    for method in fs_methods:
        print("Selecting Features with {}".format(method))
        features_selected[method] = feature_selector_method[method](
            df, label, k)
    return features_selected


def heuristic_features_selector(df, label):
    features_selected = {}
    print("Selecting heuristic based features...")
    for method in fs_methods_heuristics:
        print("Selecting Features with {}".format(method))
        features_selected[method] = heuristic_feature_selector_method[method](
            df, label)
    return features_selected

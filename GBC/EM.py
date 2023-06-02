import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_regression, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from scipy.stats import kendalltau
# from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier


def variance_selection(df_norm, label, k=-1):
    thresholder = VarianceThreshold(threshold=.01)
    transformed_df = thresholder.fit_transform(df_norm)

    # Get variances and create pandas series with feature names as index
    variances = pd.Series(thresholder.variances_, index=df_norm.columns)

    # Sort the series by variances in descending order
    sorted_features = variances.sort_values(ascending=False)

    return sorted_features


def mutual_info_selection(df_norm, label, k=-1):
    # Compute mutual information scores
    scores = mutual_info_classif(df_norm, label)

    # Create a Pandas Series with feature names as index and scores as values
    feature_scores = pd.Series(scores, index=df_norm.columns)

    # Sort the features by score in descending order
    feature_scores_sorted = feature_scores.sort_values(ascending=False)

    return feature_scores_sorted


def randomforest_Selection(df_norm, label, k=-1):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(df_norm, label)
    feature_imp = pd.Series(clf.feature_importances_,
                            index=df_norm.columns).sort_values(ascending=False)
    if(k == -1):
        sorted_features = feature_imp
    else:
        sorted_features = feature_imp[:min(k, len(df_norm.columns))]
    top_k_features = list(sorted_features.index)
    return sorted_features


def logistic_regression_selection(df_norm, label, k=-1):
    logreg = LogisticRegression(
        C=1, penalty='l2', max_iter=10000, solver='liblinear', multi_class='auto')
    logreg.fit(df_norm, label)
    coef_abs = np.abs(logreg.coef_[0])
    feature_scores = pd.Series(
        coef_abs, index=df_norm.columns).sort_values(ascending=False)
    if(k == -1):
        sorted_scores = feature_scores
    else:
        sorted_scores = feature_scores[:min(k, len(df_norm.columns))][:k]
    return sorted_scores


def corr_selection(df_norm, label, k=-1):
    featureTargetCorr = []
    for col in df_norm:
        featureTargetCorr.append(
            pearsonr(df_norm[col].astype('float64'), label.astype('float64'))[0])
    score = pd.Series(np.absolute(featureTargetCorr))
    score.index = df_norm.columns
    score.sort_values(ascending=False, inplace=True)
    if (k == -1):
        selected_features = score
    else:
        selected_features = score[:min(k, len(df_norm.columns))]
    top_k_features = list(selected_features.index)
    return selected_features


# mutual_info_selection,
#               logistic_regression_selection, kendall_selection, chi2_selection,
fs_methods = [mutual_info_selection, variance_selection,
              logistic_regression_selection, randomforest_Selection, corr_selection]

# List of feature selection methods used. These methods returns a list of top features in ascending order.


def create_features_list(res):
    sorted_features_list = []

    for i in range(len(res)):
        sorted_features = res.index[0:i+1].tolist()
        sorted_features_list.append(sorted_features)

#     print(sorted_features_list)
    return sorted_features_list


def ensemble_fs(df_norm, label):
    scores = []
    normalized_scores = []

#     Normalizing the scores of features obtained by each feature selection method.
    for fs in fs_methods:
        scores = fs(df_norm, label, -1)
        normalized_scores.append(
            ((scores - np.min(scores)) / (np.max(scores) - np.min(scores))).sort_index())

#     Putting the normalized scores of different fs methods in a dataframe to find the mean of each features.
    data = pd.concat(normalized_scores, axis=1)
    aggregate = data.mean(axis=1)
    result = pd.Series(data=aggregate.values,
                       index=aggregate.index).sort_values(ascending=False)

    sorted_features_list = create_features_list(result)
    return sorted_features_list

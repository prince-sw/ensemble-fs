import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, RFE, SelectFromModel, VarianceThreshold
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from feature_selection.unsupervised import *
from feature_selection.Ensemble import ensemble_fs

# 1


def randomforest_Selection(df_norm, label, k):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(df_norm, label)
    feature_imp = pd.Series(clf.feature_importances_,
                            index=df_norm.columns).sort_values(ascending=False)
    # selected_col = []
    # for i in range(len(df_norm.columns)):
    #     if clf.feature_importances_[i] >= 0.02:
    #         selected_col.append(df_norm.columns[i])

    selected_col = feature_imp[:min(k, len(df_norm.columns))].index
    return list(selected_col)

# 2


def chitest_selection(df_norm, label, k):
    X_new = SelectKBest(k=min(k, len(df_norm.columns)), score_func=chi2)
    z = X_new.fit_transform(df_norm, label)
    filter = X_new.get_support()
    Features = np.array(df_norm.columns)
    return Features[filter]

# 3


def Mutualinfo_selection(df_norm, label, k):
    mi = mutual_info_regression(df_norm, label)
    mi = pd.Series(mi)
    mi.index = df_norm.columns
    mi.sort_values(ascending=False, inplace=True)
    # selected_col = []
    # for i in range(len(df_norm.columns)):
    #     if mi.values[i] <= 0.1:
    #         selected_col.append(df_norm.columns[i])
    selected_col = mi[:min(k, len(df_norm.columns))].index
    return list(selected_col)

# 4


def Lasso_selection(df_norm, label, k):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    search = GridSearchCV(pipeline,
                          {'model__alpha': np.arange(0.1, 10, 0.1)},
                          cv=20, scoring="neg_mean_squared_error", verbose=0)

    search.fit(df_norm, label)
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    score = pd.Series(importance)
    score.index = df_norm.columns
    score.sort_values(ascending=False, inplace=True)
    return list(score[:min(k, len(df_norm.columns))].index)

# 5


def boruta_selection(df_norm, label, k):
    X_train, X_test, y_train, y_test = train_test_split(
        df_norm, label, test_size=.2, random_state=42)
    rfc = RandomForestClassifier(random_state=42, n_estimators=50)
    boruta_selector = BorutaPy(
        rfc, n_estimators='auto', verbose=0, random_state=42)
    boruta_selector.fit(np.array(X_train), np.array(y_train))
    selected_rf_features = pd.DataFrame({'Feature': list(X_train.columns),
                                         'Ranking': boruta_selector.ranking_})
    selected_rf_features.sort_values(by='Ranking')
    score = pd.Series(selected_rf_features.Ranking)
    score.index = selected_rf_features.Feature
    score.sort_values(ascending=True, inplace=True)
    return list(score[:min(k, len(df_norm.columns))].index)

# 6


def rfe_selection(df_norm, label, k):
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=k)
    X_new = rfe.fit_transform(df_norm, label)
    Features = np.array(df_norm.columns)
    return Features[rfe.support_ == True]

# 7


def logistic_regression_selection(df_norm, label, k):
    sel_ = SelectFromModel(LogisticRegression(
        C=1, penalty='l2', max_iter=10000), max_features=k, threshold=-np.inf)
    sel_.fit(df_norm, label)
    sel_.get_support()
    selected_feat = df_norm.columns[(sel_.get_support())]
    return list(selected_feat)

# 8


def variance_selection(df_norm, label, k):
    thresholder = VarianceThreshold(threshold=.03)
    thresholder.fit_transform(df_norm)
    score = pd.Series(thresholder.variances_)
    score.index = df_norm.columns
    score.sort_values(ascending=False, inplace=True)
    return list(score[:min(k, len(df_norm.columns))].index)

# 9


def treebased_selection(df_norm, label, k):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(df_norm, label)
    model = SelectFromModel(
        clf, prefit=True, max_features=k, threshold=-np.inf)
    selected_feat = df_norm.columns[(model.get_support())]
    return list(selected_feat)

# 10


def corr_selection(df_norm, label, k):
    featureTargetCorr = []
    for col in df_norm:
        featureTargetCorr.append(
            pearsonr(df_norm[col].astype('float64'), label.astype('float64'))[0])
    score = pd.Series(np.absolute(featureTargetCorr))
    score.index = df_norm.columns
    score.sort_values(ascending=False, inplace=True)
    return list(score[:min(k, len(df_norm.columns))].index)


def ensemble_selection(df_norm, label, k):
    return ensemble_fs(df_norm, label, k)

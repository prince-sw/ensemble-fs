from sklearn.model_selection import cross_validate
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score, roc_auc_score


scoring = {
    "accuracy": make_scorer(accuracy_score),
    "fmeasure": make_scorer(f1_score, average="weighted", zero_division=0),
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    "roc": make_scorer(roc_auc_score, average="weighted", multi_class="ovr", needs_proba=True)
}

def get_result(df, target):
    model_dict = get_models()
    results = {}
    for model in model_dict.keys():
        result = cross_validate(estimator=model_dict[model], X=df.drop(
            target, axis=1), y=df[target], cv=10, scoring=scoring, return_train_score=True, verbose=0)
        print("Fitting done with {}".format(model))
        results[model] = result
    return results


def get_models():
    model1 = LogisticRegression(solver='liblinear')
    model2 = GaussianNB()
    model3 = KNeighborsClassifier()
    model4 = RandomForestClassifier()
    model5 = DecisionTreeClassifier()
    model6 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model7 = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
    model8 = svm.SVC(kernel='linear', C=1, probability=True)
    model9 = CalibratedClassifierCV(RidgeClassifier())
    model10 = XGBClassifier()





    model_dict = {
        "lr": model1,
        "nb": model2,
        "knn": model3,
        "rf": model4,
        "dt": model5,
        "gb": model6,
        "ab": model7,
        "svc": model8,
        "ri": model9,
        "xgb": model10,
    }
    return model_dict

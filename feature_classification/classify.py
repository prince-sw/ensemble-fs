import pandas as pd
import numpy as np
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score, roc_auc_score
from datasets.data import clear_missing, encode_cols, drop_cols, scale_dataset
from feature_selection.feature_selector import feature_selector
from feature_selection.feature_selector import heuristic_features_selector
from observations.model_config import fs_methods, models

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "fmeasure": make_scorer(f1_score, average="weighted", zero_division=0),
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    "roc": make_scorer(roc_auc_score, average="weighted", multi_class="ovr", needs_proba=True)
}


def get_score(results):
    ans = "{},{},{},{},{}\n".format(results["test_accuracy"].mean(),
                                    results["test_fmeasure"].mean(),
                                    results["test_precision"].mean(),
                                    results["test_recall"].mean(),
                                    results["test_roc"].mean())
    # ans = "{}\n".format(results["test_accuracy"].mean())
    return ans


def write_results(file, results, fs_method, k):
    for result in results.keys():
        file.write(result+","+fs_method+","+str(k) +
                   ","+get_score(results[result]))


def get_result(df, target, is_multiclass):
    model1 = LogisticRegression(max_iter=1000000)
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
    results = {}
    for model in models:
        result = cross_validate(estimator=model_dict[model], X=df.drop(
            target, axis=1), y=df[target], cv=5, scoring=scoring, return_train_score=True, verbose=0)
        print("Fitting done with {}".format(model))
        results[model] = result
    return results


def get_result_split(df, target, is_multiclass):
    model1 = None
    if is_multiclass:
        model1 = LogisticRegression(
            max_iter=1000000, multi_class="multinomial")
    else:
        model1 = LogisticRegression(max_iter=1000000)
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
    results = {}
    X = df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    for model in models:
        # change scoring method
        model_dict[model].fit(x_train, y_train)
        y_pred = model_dict[model].predict(x_test)
        acc = accuracy_score(y_test, y_pred, normalize=True)
        result = {"test_accuracy": acc}
        # result = cross_validate(estimator=model_dict[model], X=df.drop(
        #     target, axis=1), y=df[target], cv=5, scoring=scoring, return_train_score=True, verbose=0)
        print("Fitting done with {}".format(model))
        results[model] = result
    return results


def classify_dataset(dataset):
    # read data
    print("Reading dataset {}.csv...".format(dataset["name"]))
    dataset_path = "./datasets/"+dataset["name"]+".csv"
    df = pd.read_csv(dataset_path, sep=dataset["sep"])
    print("Reading finished... \n")

    # preprocessing
    print("Starting preprocessing... ")
    df = drop_cols(df, dataset["drop_columns"])
    df = encode_cols(df, dataset["encode_columns"])
    df = clear_missing(df)
    df = scale_dataset(df, dataset["no_scale"])
    print("Finished preprocessing... \n")

    # classification
    file = open("./results/{}.txt".format(dataset["name"]), "w")
    # file.write("model,fs_method,k,accuracy,fmeasure,precision,recall,roc\n")
    file.write("model,fs_method,k,accuracy,fmeasure,precision,recall,roc\n")
    print("Training models...")
    # get results for complete dataset
    print("Training for all columns")
    results = get_result(df, dataset["target"], dataset['is_multiclass'])
    print("Finished training... ")
    print("Writing Results... ")
    write_results(file, results, "nofs", k=len(df.columns)-1)
    print("Finished writing Results... ")

    # run for every number of columns chosen
    for k in range(1, len(df.columns)+1):
        # get selected features using every method for one k
        selected_features = feature_selector(
            df.drop(dataset["target"], axis=1), df[dataset["target"]], k)

        # train for every selected features from each method and save that result
        for method in fs_methods:
            # get the selected df columns with target
            to_take = list(selected_features[method])
            to_take.append(dataset["target"])
            selected_df = df[to_take]

            results = get_result(
                selected_df, dataset["target"], dataset['is_multiclass'])
            # write result for that k for that method
            print(f"Writing Results for {method} and k={k}... ")
            write_results(file, results, method, k=k)
            print("Finished writing Results... ")

    # print("Starting with heuristic optimizers..")
    # selected_features = heuristic_features_selector(
    #     df.drop(dataset["target"], axis=1), df[dataset["target"]])
    # for method in selected_features.keys():
    #     # get the selected df columns with target
    #     to_take = list(selected_features[method])
    #     to_take.append(dataset["target"])
    #     selected_df = df[to_take]

    #     results = get_result(
    #         selected_df, dataset["target"], dataset['is_multiclass'])
    #     # write result for that k for that method
    #     print(f"Writing Results for {method} and k={len(to_take)-1}... ")
    #     write_results(file, results, method, k=len(to_take)-1)
    #     print("Finished writing Results... ")

    file.close()
    print("Done with dataset {}.csv...\n".format(dataset["name"]))


def get_tot_acc(results):
    acc = 0
    for result in results.keys():
        acc += result["test_accuracy"].mean()
    return acc

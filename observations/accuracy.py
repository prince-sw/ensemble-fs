import matplotlib.pyplot as plt
import pandas as pd
from datasets.config import data_files
from observations.model_config import fs_methods, models


def save_figs(k, scores, fs_method, name):
    print("Plotting for {},{}".format(name, fs_method))
    plt.figure(figsize=(14, 5))
    for model in models:
        plt.plot(range(1, k+1, 4), scores[model][::4], label=model)
    plt.title("{}//{}".format(name, fs_method))
    plt.xlabel("No. of features")
    plt.xticks(range(1, k+1, 4))
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.savefig('./plots/graphs/{}_{}.png'.format(name, fs_method))
    plt.close()


def save_accuracy_table_entry(file, dataset):
    df = pd.read_csv("./results/{}.txt".format(dataset["name"]))
    row = df.query("accuracy == accuracy.max()").query("k == k.min()")
    write_str = "{},{},{},{},{},{},{},{},{}\n".format(dataset["name"],
                                                      row["model"].iloc[0],
                                                      row["fs_method"].iloc[0],
                                                      row["k"].iloc[0],
                                                      row["accuracy"].iloc[0],
                                                      row["fmeasure"].iloc[0],
                                                      row["precision"].iloc[0],
                                                      row["recall"].iloc[0],
                                                      row["roc"].iloc[0])
    # write_str = "{},{},{},{},{}\n".format(dataset["name"],
    #                                       row["model"].iloc[0],
    #                                       row["fs_method"].iloc[0],
    #                                       row["k"].iloc[0],
    #                                       row["accuracy"].iloc[0]), row["fmeasure"].iloc[0], row["precision"].iloc[0], row["recall"].iloc[0],                                           row["roc"].iloc[0])
    file.write(write_str)


def save_acc_graphs():
    for dataset in data_files:
        print("Calculating scores for {}".format(dataset["name"]))
        df = pd.read_csv("./results/{}.txt".format(dataset["name"]))
        print(df.head())
        columns = df["k"].max()
        print(columns)
        for fs_method in fs_methods:

            scores = {
                "lr": [],
                "nb": [],
                "knn": [],
                "rf": [],
                "dt": [],
                "gb": [],
                "ab": [],
                "svc": [],
                "ri": [],
                "xgb": []
            }
            for i in range(1, columns):
                for model in models:
                    q = "fs_method=='{}' and model=='{}' and k=={}".format(
                        fs_method, model, i)
                    acc = df.query(q)["accuracy"].iloc[0]
                    scores[model].append(acc)
            for model in models:
                q = "fs_method=='{}' and model=='{}' and k=={}".format(
                    fs_method, model, i)
                acc = df.query(q)["accuracy"].iloc[0]
                scores[model].append(acc)
            save_figs(columns, scores, fs_method, dataset["name"])


def save_acc_table():
    print("Saving Accuracy Table...")
    file = open("./tables/acc_table.txt", "w")
    file.write(
        "dataset,model,fs_method,k,accuracy,fmeasure,precision,recall,roc\n")
    for dataset in data_files:
        save_accuracy_table_entry(file, dataset)
    file.close()

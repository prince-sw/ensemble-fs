import pandas as pd
from PreProcessing import *
from EM import ensemble_fs
from Classify import get_result
from Result import write_results

dataset = {
    "name": "gbc_org",
    "drop_columns": [],
    "encode_columns": ['Label'],
    "target": "Label",
    "no_scale":['Label'],
}

def run():
    df = pd.read_csv(dataset['name']+'.csv')
    df = drop_cols(df, dataset["drop_columns"])
    df = encode_cols(df, dataset["encode_columns"])
    df = clear_missing(df)
    df = scale_dataset(df, dataset["no_scale"])

    ranking = ensemble_fs(df.drop(dataset["target"], axis=1), df[dataset["target"]])

    file = open("{}_results.csv".format(dataset["name"]), "w")
    file.write("model,k,accuracy,fmeasure,precision,recall,roc\n\n")


    for i in range(len(df.columns)-1):
        to_take = list(ranking[i])
        to_take.append(dataset["target"])
        selected_df = df[to_take]
        # print(selected_df)
        # print()

        results = get_result(selected_df, dataset["target"])
        print(f"Writing Results for and k={i+1}... ")
        write_results(file, results, i+1)
    print("Finished writing Results... ")




    

if __name__ == '__main__':
    run()

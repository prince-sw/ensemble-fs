from datasets.config import data_files
from feature_classification.classify import classify_dataset
from observations.accuracy import save_acc_graphs, save_acc_table, save_fs_table
from observations.heatmap import plot_heatmaps
from tqdm import tqdm

if __name__ == "__main__":
    for i in tqdm(range(1, 21), desc="Iteration: "):
        for dataset in data_files:
            classify_dataset(dataset, i)
            # pass
        # print("Plotting Accuracy Graphs")
        # save_acc_graphs()
        # save_acc_table()
        # save_fs_table(i)
    # plot_heatmaps()
    # print("Exiting...\n")

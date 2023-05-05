from datasets.config import data_files
from feature_classification.classify import classify_dataset
from observations.accuracy import save_acc_graphs, save_acc_table
from observations.heatmap import plot_heatmaps

if __name__ == "__main__":
    for dataset in data_files:
        classify_dataset(dataset)
        pass
    print("Plotting Accuracy Graphs")
    save_acc_graphs()
    save_acc_table()
    plot_heatmaps()
    print("Exiting...\n")

from feature_selection.CosineSimilarity import CosineSimilarity
from feature_selection.LaplacianScore import LaplacianScore
from feature_selection.IterativeLaplacianScore import IterativeLaplacianScore
from FRUFS import FRUFS


def laplacian_selection(df, k):
    ls = LaplacianScore(df, k=k)
    ls.calculate_scores()
    lsfs = ls.selectKBest()
    return lsfs


def cosine_selection(df, k):
    cs = CosineSimilarity(df, k=k)
    cs.calculate_scores()
    csfs = cs.selectKBest()
    return csfs


def iterlaplacian_selection(df, k):
    ils = IterativeLaplacianScore(df, k=k)
    ils.calculate_scores()
    ilsfs = ils.selectKBest()
    return ilsfs


def pairwise_corr_selection(df, k):
    corr = df.corr(method='pearson')
    removed_columns = []
    threshold = 0.75
    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i, j] >= threshold:
                removed_columns.append(corr.columns[i])
    removed_columns = list(set(removed_columns))
    selected_cols = [x for x in df.columns if x not in removed_columns]
    if len(selected_cols) < k:
        selected_cols.extend(removed_columns[:(k-len(selected_cols))])
    return selected_cols[:min(k, len(df.columns))]


def frufs_selection(df, k):
    model = FRUFS(verbose=0, k=k)
    model.fit(df)
    return list(model.columns_[:min(k, len(df.columns))])

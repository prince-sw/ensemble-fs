import pandas as pd
import numpy as np
from skfeature.utility.construct_W import construct_W


class IterativeLaplacianScore:
    def __init__(self, df=pd.DataFrame([[]]), k=1, s=1):
        try:
            self.df = pd.DataFrame(df)
            self.col_scores = [0]*len(df.columns)
            self.k = k
            self.s = s
            self.col_score_dict = dict(zip(self.df.columns, self.col_scores))
            assert not self.df.isnull().values.any()
        except AssertionError as a:
            print("Please clean the data from NaN Values")

    def calculate_laplacian(self, df):
        X = df.to_numpy()
        ROWS = len(X)
        ones = np.ones(ROWS).reshape(ROWS, 1)

        S = construct_W(X)
        D = np.diag(np.array(np.matmul(S.todense(), ones)).reshape(-1))
        L = np.array(D-S.todense())

        def getMean(f):
            return ((np.matmul(np.matmul(np.transpose(f), D), ones)) / (np.matmul(np.matmul(np.transpose(ones), D), ones))).reshape(-1)[0]

        Fr = [df[f].to_numpy().reshape(ROWS, 1) for f in df]
        Frt = [(f-getMean(f)*ones) for f in Fr]

        def getLaplacianScore(f):
            num = (np.matmul(np.matmul(np.transpose(f), L), f).reshape(-1)[0])
            den = (np.matmul(np.matmul(np.transpose(f), D), f).reshape(-1)[0])
            if den == 0:
                den = 10000
            return num/den

        col_scores = [getLaplacianScore(f) for f in Frt]
        colscores = dict(zip(df.columns, col_scores))
        for key in colscores.keys():
            self.col_score_dict[key] = colscores[key]
        col_names = [qq[0] for qq in sorted(colscores.items(), key=lambda x:-x[1])[
            :min(self.s, len(self.df.columns))]]
        # returns names of lowest s columns to drop based on score
        return col_names

    def calculate_scores(self):
        current_df = self.df.copy()
        chosen_cols = np.array(current_df.columns)

        while len(chosen_cols) > self.k:
            cols_to_delete = self.calculate_laplacian(current_df)
            chosen_cols = [
                x for x in current_df.columns if x not in cols_to_delete]
            current_df = current_df[chosen_cols]

        self.col_scores = [self.col_score_dict[x] for x in self.df.columns]

    def selectKBest(self):
        col_names = [qq[0] for qq in sorted(self.col_score_dict.items(), key=lambda x:x[1])[
            :min(self.k, len(self.df.columns))]]
        return col_names

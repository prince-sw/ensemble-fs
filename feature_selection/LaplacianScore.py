import pandas as pd
import numpy as np
from skfeature.utility.construct_W import construct_W


class LaplacianScore:
    def __init__(self, df=pd.DataFrame([[]]), k=1):
        try:
            self.df = pd.DataFrame(df)
            self.col_scores = [0]*len(df.columns)
            self.k = k
            assert not self.df.isnull().values.any()
        except AssertionError as a:
            print("Please clean the data from NaN Values")

    def calculate_scores(self):
        df = self.df
        X = self.df.to_numpy()
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

        col_score = [getLaplacianScore(f) for f in Frt]
        self.col_scores = col_score

    def selectKBest(self):
        colscores = dict(zip(self.df.columns, self.col_scores))
        col_names = [qq[0] for qq in sorted(colscores.items(), key=lambda x:x[1])[
            :min(self.k, len(self.df.columns))]]
        return col_names

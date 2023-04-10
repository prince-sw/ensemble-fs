import numpy as np
import pandas as pd
from sklearn import preprocessing
from numpy.linalg import norm
import math


class CosineSimilarity:
    def __init__(self, df=pd.DataFrame([[]]), k=1, scoring='exp'):
        try:
            self.df = pd.DataFrame(df)
            self.col_scores = [0]*len(df.columns)
            self.k = k
            self.scoring = scoring
            assert not self.df.isnull().values.any()
        except AssertionError as a:
            print("Please clean the data from NaN Values")

    def calculate_scores(self):
        df = self.df
        n = len(df.columns)  # no. of features.
        discern_arr = np.ones(n)
        for i in range(n):
            discern_arr[i] = df.iloc[:, i].std()

        cosineS_matrix = np.ones([n, n])
        for i in range(n):
            for j in range(n):
                cosineS_matrix[i, j] = np.dot(
                    df.iloc[:, i], df.iloc[:, j])/(norm(df.iloc[:, i])*norm(df.iloc[:, j]))

        mdv = max(discern_arr)  # max discernibility value
        idx_mdv = np.where(discern_arr == mdv)[0][0]  # index no. of mdv.

        ind_arr = self.getIndependence(n, idx_mdv, cosineS_matrix, discern_arr)
        col_score = np.array([ind_arr[i]*discern_arr[i] for i in range(n)])
        self.col_scores = col_score

    def selectKBest(self):
        colscores = dict(zip(self.df.columns, self.col_scores))
        col_names = [qq[0] for qq in sorted(colscores.items(), key=lambda x:-x[1])[
            :min(self.k, len(self.df.columns))]]
        return col_names

    def getIndependence(self, n, idx_mdv, cosineS_matrix, discern_arr):

        if self.scoring == 'rec':
            rec_ind_arr = np.ones(n)

            for i in range(n):
                if i == idx_mdv:
                    rec_ind_arr[i] = (max(1/cosineS_matrix[0]))
                else:
                    rec_ind_arr[i] = (
                        min(1/cosineS_matrix[0][discern_arr > discern_arr[i]]))
            return rec_ind_arr

        elif self.scoring == 'ant':
            anti_ind_arr = np.ones(n)

            for i in range(n):
                if i == idx_mdv:
                    anti_ind_arr[i] = (max(1-cosineS_matrix[0]))
                else:
                    anti_ind_arr[i] = (
                        min(1-cosineS_matrix[0][discern_arr > discern_arr[i]]))
            return anti_ind_arr

        else:
            exp_ind_arr = np.ones(n)

            for i in range(n):
                if i == idx_mdv:
                    exp_ind_arr[i] = math.exp(max(-cosineS_matrix[0]))
                else:
                    exp_ind_arr[i] = math.exp(
                        min(-cosineS_matrix[0][discern_arr > discern_arr[i]]))
            return exp_ind_arr

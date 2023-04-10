from feature_selection.GeneticAlgoritm import geneticAlgo
from feature_selection.ParticleSwarmOptimization import funcPSO
from feature_selection.MantaRayForaging import runmrf
from feature_selection.GoldenRatioOptimization import goldenratiomethod
from feature_selection.SocialMimicOptimizer import socialmimic


def genetic_algorithm(df, label):
    best_cols = geneticAlgo(df, label)
    return list(best_cols)


def particleswarm_algorithm(df, label):
    best_cols = funcPSO(df, label)
    return list(best_cols)


def mantaray_optimization(df, label):
    best_cols = runmrf(df, label)
    return list(best_cols)


def goldenratio_optimizer(df, label):
    best_cols = goldenratiomethod(df, label)
    return list(best_cols)


def socialmimic_optimizer(df, label):
    best_cols = socialmimic(df, label)
    return list(best_cols)

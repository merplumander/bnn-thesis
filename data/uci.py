# Data sets and splits taken from https://github.com/yaringal/DropoutUncertaintyExps

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston as sklearn_load_boston
from sklearn.model_selection import train_test_split


def load_uci_data(dataset, validation_split=0.0):
    data = np.loadtxt(f"uci_data/{dataset}/data/data.txt")
    index_features = np.loadtxt(f"uci_data/{dataset}/data/index_features.txt").astype(
        np.int
    )
    index_target = np.loadtxt(f"uci_data/{dataset}/data/index_target.txt").astype(
        np.int
    )
    x = data[:, index_features].astype(np.float32)
    y = data[:, index_target].astype(np.float32)
    if dataset == "boston":
        assert x.shape == (506, 13)
    if dataset == "concrete":
        assert x.shape == (1030, 8)
    if dataset == "energy":
        assert x.shape == (768, 8)
    if dataset == "kin8nm":
        assert x.shape == (8192, 8)
    if dataset == "naval":
        assert x.shape == (11934, 16)
    if dataset == "power":
        assert x.shape == (9568, 4)
    if dataset == "protein":
        assert x.shape == (45730, 9)
    if dataset == "wine":
        assert x.shape == (1599, 11)
    if dataset == "yacht":
        assert x.shape == (308, 6)
    n_splits = np.loadtxt(f"uci_data/{dataset}/data/n_splits.txt").astype(np.int)
    train_indices = []
    validation_indices = []
    test_indices = []
    for i in range(n_splits):
        train = np.loadtxt(f"uci_data/{dataset}/data/index_train_{i}.txt").astype(
            np.int
        )
        if validation_split:
            train, validation = train_test_split(
                train, test_size=validation_split, random_state=0
            )
            assert len(set(train).intersection(set(validation))) == 0
            validation_indices.append(validation)
        else:
            validation_indices.append(np.array([], dtype=np.int))
        train_indices.append(train)
        test = np.loadtxt(f"uci_data/{dataset}/data/index_test_{i}.txt").astype(np.int)
        test_indices.append(test)
        assert len(set(train).intersection(set(test))) == 0
    return x, y, train_indices, validation_indices, test_indices


# def load_boston():
#     x = sklearn_load_boston()["data"].astype(np.float32)
#     y = sklearn_load_boston()["target"].astype(np.float32)
#     assert x.shape == (506, 13)
#     return x, y
#
#
# def load_concrete():
#     data = pd.read_excel("uci_data/Concrete_Data.xls")
#     x = data.iloc[:, :-1].to_numpy().astype(np.float32)
#     y = data.iloc[:, -1].to_numpy().reshape(-1, 1).astype(np.float32)
#     assert x.shape == (1030, 8)
#     return x, y
#
#
# def load_energy():
#     """
#     This dataset has two output values. It is not clear which one to use or whether to
#     use both. Gal 2015 seems to have used only the first output value
#     (https://github.com/yaringal/DropoutUncertaintyExps) and claims that
#     Hernandez-Lobato did the same.
#     """
#     df = pd.read_excel("uci_data/ENB2012_data.xlsx")
#     x = df.iloc[:, :-2].to_numpy().astype(np.float32)
#     y = df.iloc[:, -2].to_numpy().reshape(-1, 1).astype(np.float32)
#     assert x.shape == (768, 8)
#     assert y.shape == (768, 1)
#     print(
#         "Reminder: Check for this dataset (energy efficiency) whether to use one or two output variables"
#     )
#     return x, y
#
#
# def load_kin8nm():
#     df = pd.read_csv("uci_data/dataset_2175_kin8nm.csv")
#     x = df.iloc[:, :-1].to_numpy().astype(np.float32)
#     y = df.iloc[:, -1].to_numpy().reshape(-1, 1).astype(np.float32)
#     assert x.shape == (8192, 8)
#     return x, y

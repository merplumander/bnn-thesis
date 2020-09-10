# Data sets and splits taken from https://github.com/yaringal/DropoutUncertaintyExps

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston as sklearn_load_boston
from sklearn.model_selection import train_test_split


def _load_uci_dataset(dataset, validation_split=0.0):
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
    return x, y


def load_uci_data(dataset, validation_split=0.0, gap_data=False):
    x, y = _load_uci_dataset(dataset, validation_split)
    data_path = "uci_gap_data" if gap_data else "uci_data"
    train_split_path = "train_indices_" if gap_data else "index_train_"
    test_split_path = "test_indices" if gap_data else "index_test_"
    n_splits = np.loadtxt(f"{data_path}/{dataset}/data/n_splits.txt").astype(np.int)
    train_indices = []
    validation_indices = []
    test_indices = []
    for i in range(n_splits):
        train = np.loadtxt(
            f"{data_path}/{dataset}/data/{train_split_path}{i}.txt"
        ).astype(np.int)
        if validation_split:
            train, validation = train_test_split(
                train, test_size=validation_split, random_state=0
            )
            assert len(set(train).intersection(set(validation))) == 0
            validation_indices.append(validation)
        else:
            validation_indices.append(np.array([], dtype=np.int))
        train_indices.append(train)
        test = np.loadtxt(
            f"{data_path}/{dataset}/data/{test_split_path}{i}.txt"
        ).astype(np.int)
        test_indices.append(test)
        assert len(set(train).intersection(set(test))) == 0
    return x, y, train_indices, validation_indices, test_indices

import pandas as pd
import numpy as np
import h5py
import torch
from tqdm import tqdm
from scipy.stats import skew
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
from imblearn.over_sampling import SMOTE


def load_data(data_path, label=None, protein_name_list=None, sample_size=None, features_list=None, mode=None, conformation=None):
    input_fo = h5py.File(data_path, 'r')

    X = np.ndarray([], dtype=np.float32)
    y = np.ndarray([], dtype=np.float32)
    i = 0

    if protein_name_list is None:
        protein_name_list = list(input_fo.keys())
    print("loading", len(protein_name_list), "proteins.")
    for protein_name in tqdm(protein_name_list):
        x_, y_ = load_protein(data_path, label=label, protein_name=protein_name, sample_size=sample_size,
                              features_list=features_list, mode=mode, conformation=conformation)
        if i == 0:
            X = x_.astype(np.float32)
            y = y_.astype(np.float32)
        else:
            X = np.vstack((X, x_.astype(np.float32)))
            y = np.vstack((y, y_.astype(np.float32)))
        i += 1

    return StandardScaler().fit_transform(Imputer().fit_transform(X)), y


def load_protein(data_path, label=None, protein_name=None, sample_size=None, features_list=None, mode=None, conformation=None):
    input_fo = h5py.File(data_path, 'r')
    if label is None:
        label = "label"
    # if features_list is none then use all of the features
    if features_list is None:
        features_list = list(input_fo[str(protein_name)].keys())
        if label in features_list:
            features_list.remove(label)
        if "receptor" in features_list:
            features_list.remove("receptor")
        if "drugID" in features_list:
            features_list.remove("drugID")
        if "label" in features_list:
            features_list.remove("label")

        # in order to determine indices, select all of the labels and conformations, then seperately choose based on specifiedconditions, then find the intersection of the two sets.
    full_labels = np.asarray(input_fo[str(protein_name)][label]).flatten()
    full_idxs = np.arange(0, full_labels.shape[0], 1)

    mode_idxs = []
    if mode is not None:
        mode_idxs = full_idxs[full_labels[:, ] == mode]
    else:
        mode_idxs = full_idxs

    full_idxs = np.intersect1d(mode_idxs, full_idxs)

    # if sample size is none then select all of the indices
    if sample_size is None or sample_size > len(full_idxs):
        sample_size = len(full_idxs)

    sample = np.sort(np.random.choice(full_idxs, sample_size, replace=False))

    # get the data and store in numpy array
    data_array = np.zeros([sample_size, len(features_list)])
    i = 0

    for dataset in features_list:
        data = np.asarray(input_fo[str(protein_name)][str(dataset)], dtype=np.float32)[sample]
        data_array[:, i] = data[:, 0]
        i += 1

    label_array = np.asarray(input_fo[str(protein_name)][label])[sample]

    return data_array.astype(np.float32), label_array.astype(np.float32)


class KinaseDataset(Dataset):

    def __init__(self, data_path, label=None, protein_name_list=None, sample_size=None, features_list=None, mode=None):
        self.data, self.labels = load_data(data_path=data_path, label=label, protein_name_list=protein_name_list, sample_size=sample_size,
                              features_list=features_list, mode=mode)
        # is there a more efficient way to do this??
        self.data, self.labels = SMOTE(ratio="minority").fit_sample(self.data,self.labels.flatten())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # return torch.from_numpy(self.data[item]), torch.from_numpy(self.labels[item])
        return self.data[item], self.labels[item].flatten()

def parse_features(feature_path, null_path=None):
    with open(feature_path, "r") as input_file:
        feature_list = []
        for line in input_file:
            line = line.strip('\n')
            feature_list.append(line)

        if null_path is not None:
            with open(null_path, "r") as null_input_file:
                for line in null_input_file:
                    line = line.strip('\n')
                    if line in feature_list:
                        feature_list.remove(line)
            return feature_list
        else:
            return feature_list

def test_kinase_dataset(data_path="/Users/derekjones2025/workspace/protein_binding/data/all_kinase/with_pocket/full_kinase_set.h5",
                        feature_path="/Users/derekjones2025/workspace/protein_binding/data/all_kinase/with_pocket/binding_features_list.csv",
                        null_path="/Users/derekjones2025/workspace/protein_binding/data/all_kinase/with_pocket/null_column_list.csv",
                        protein_name_list=["lck"]):
    features_list = parse_features(feature_path=feature_path,null_path=null_path)
    data_obj = KinaseDataset(data_path,protein_name_list=protein_name_list, features_list=features_list)

    return data_obj

# X = test_kinase_dataset()
# print(X)

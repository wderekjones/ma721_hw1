import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from input_pipeline import KinaseDataset, parse_features
from torch.autograd import Variable
from model import Net


parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, help="path to data")
parser.add_argument("--feats", type=str, help="path to features")
parser.add_argument("--null", type=str, help="path to null features")
parser.add_argument("--model", type=str, help="path to model")
args = parser.parse_args()

features_list = parse_features(args.feats, null_path=args.null)

data = KinaseDataset(data_path=args.data, oversample=False,split="test", features_list=features_list, protein_name_list=["lck"])

N, D_in, H, D_out = 0, data.data.shape[1], 5, 2


model = Net(D_in=D_in, H=H, D_out=D_out, N=N, p=0.7)
model.load_state_dict(torch.load(args.model))
model.cuda()

print("performance on test set: ")
test_y_probs = model(Variable(data.data.cuda(),requires_grad=False))
test_y_preds = np.argmax(test_y_probs.data.numpy(),axis=1)
y_test = np.argmax(data.labels.data.numpy(),axis=1)

print("Accuracy: {} \t F1-Score: {} \t Precision: {} \t Recall: {}".format(accuracy_score(y_test,test_y_preds),f1_score(y_test,test_y_preds),
                                                                               precision_score(y_test,test_y_preds),
                                                                               recall_score(y_test,test_y_preds)))

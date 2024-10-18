from model import models
from dataset import JSONL_Dataset
from util import *

ds = JSONL_Dataset(['./data1', './data2', './data3'])
num_models = 8

for i in range(num_models):
    model = models[i]
    X = ds.x[i]
    Y = ds.y[i]
    train_loader, test_loader, val_loader = get_train_test_val_loader(X, Y)
    print(f"Starting Training for CNN {i+1}")
    train_model(model, train_loader, val_loader, f"CNN_{i+1}")
    print(f"Starting Testing for CNN {i+1}")
    test_model(model, test_loader)
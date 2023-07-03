import numpy as np
import argparse
import random

from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score

from CRT import CRT, TFR_Encoder, Model
from base_models import SSLDataSet, FTDataSet

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader



def self_supervised_learning(model, X, n_epoch, lr, batch_size, device, min_ratio=0.3, max_ratio=0.8):
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    model.to(device)
    model.train()

    dataset = SSLDataSet(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    
    pbar = trange(n_epoch)
    for _ in pbar:
        for batch in dataloader:
            x = batch.to(device)
            loss = model(x, ssl=True, ratio=max(min_ratio, min(max_ratio, _ / n_epoch)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        scheduler.step(_)
        pbar.set_description(str(sum(losses) / len(losses)))
    torch.save(model.to('cpu'), 'Pretrained_Model.pkl')
    
def finetuning(model, train_set, valid_set, n_epoch, lr, batch_size, device, multi_label=False):
    # multi_label: whether the classification task is a multi-label task.
    model.train()
    model.to(device)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    loss_func = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    
    for stage in range(2):
        # stage0: finetuning only classifier; stage1: finetuning whole model
        best_auc = 0
        step = 0
        if stage == 0:
            min_lr = 1e-6
            optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        else:
            min_lr = 1e-8
            optimizer = optim.Adam(model.parameters(), lr=lr/2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode = 'max', factor=0.8, min_lr=min_lr)
        pbar = trange(n_epoch)
        for _ in pbar:
            for batch_idx, batch in enumerate(train_loader):
                step += 1
                x, y = tuple(t.to(device) for t in batch)
                pred = model(x)
                loss = loss_func(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    valid_auc = test(model, valid_set, batch_size, multi_label)
                    pbar.set_description('Best Validation AUC: {:.4f} --------- AUC on this step: {:.4f}'.format(best_auc, valid_auc))
                    if valid_auc > best_auc:
                        best_auc = valid_auc
                        torch.save(model, 'Finetuned_Model.pkl')
                    scheduler.step(best_auc)
                    
def test(model, dataset, batch_size, multi_label):
    model.eval()
    testloader = DataLoader(dataset, batch_size=batch_size)

    pred_prob = []
    with torch.no_grad():
        for batch in testloader:
            x, y = tuple(t.to(device) for t in batch)
            pred = model(x)
            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])
    auc = roc_auc_score(dataset.label, pred_prob, multi_class='ovr')
    model.train()
    return auc

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", type=str2bool, default=False)
    parser.add_argument("--sl", type=str2bool, default=True)
    parser.add_argument("--load", type=str2bool, default=True)
    # all default values of parameters are for PTB-XL
    parser.add_argument("--seq_len", type=int, default=10000)
    parser.add_argument("--patch_len", type=int, default=20)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--in_dim", type=int, default=12)
    parser.add_argument("--n_classes", type=int, default=5)
    opt = parser.parse_args()
    
    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seq_len = opt.seq_len
    patch_len = opt.patch_len
    dim = opt.dim
    in_dim = opt.in_dim
    n_classes = opt.n_classes
    
    if opt.ssl:
        model = Model(seq_len, patch_len, dim, n_classes, in_dim).to(device)
        X = np.load('./dataset/har_train_all.npy')
        self_supervised_learning(model, X, 100, 1e-3, 128, device)
    if opt.load:
        model = torch.load('Pretrained_Model.pkl', map_location=device)
    else:
        model = Model(seq_len, patch_len, dim, n_classes, in_dim).to(device)
    if opt.sl:
        train_X, train_y = np.load('./dataset/har_train_all.npy'), np.load('./dataset/har_train_label.npy')
        valid_X, valid_y = np.load('./dataset/har_valid_all.npy'), np.load('./dataset/har_valid_label.npy')
        TrainSet = FTDataSet(train_X, train_y, True)
        ValidSet = FTDataSet(valid_X, valid_y, True)
        finetuning(model, TrainSet, ValidSet, 100, 1e-3, 128, device, multi_label=True)
        
    
    

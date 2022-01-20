import pickle
from gnn import gnn
import time
import numpy as np
import utils
import torch.nn as nn
import torch
import time
import os
from sklearn.metrics import roc_auc_score
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader                                     
from dataset import BaseDataset, collate_fn, UnderSampler

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 50)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--dataset", help="dataset", type=str, default = "tiny")
parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
parser.add_argument("--num_workers", help="number of workers", type=int, default = os.cpu_count())
parser.add_argument("--embedding_dim", help="node embedding dim aka number of distinct node label", type=int, default = 20)
parser.add_argument("--nhop", help="number of hops", type=int, default = 3)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--data_path", help="path to the data", type=str, default='data_processed')
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default = 'save/')
parser.add_argument("--log_dir", help="logging directory", type=str, default = 'log/')
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)
parser.add_argument("--al_scale", help="attn_loss scale", type=float, default = 1.0)
parser.add_argument("--ckpt", help="Load ckpt file", type=str, default = "")
parser.add_argument("--train_keys", help="train keys", type=str, default='train_keys.pkl')
parser.add_argument("--test_keys", help="test keys", type=str, default='test_keys.pkl')

def main(args):
    #hyper parameters
    num_epochs = args.epoch
    lr = args.lr
    ngpu = args.ngpu
    batch_size = args.batch_size
    data_path = os.path.join(args.data_path, args.dataset)
    args.train_keys = os.path.join(data_path, args.train_keys)
    args.test_keys = os.path.join(data_path, args.test_keys)
    save_dir = args.save_dir
    log_dir = args.log_dir

    #make save dir if it doesn't exist
    if not os.path.isdir(save_dir):
        os.system('mkdir ' + save_dir)
    if not os.path.isdir(log_dir):
        os.system('mkdir ' + log_dir)

    #read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
    with open (args.train_keys, 'rb') as fp:
        train_keys = pickle.load(fp)
    with open (args.test_keys, 'rb') as fp:
        test_keys = pickle.load(fp)

    #print simple statistics about dude data and pdbbind data
    print (f'Number of train data: {len(train_keys)}')
    print (f'Number of test data: {len(test_keys)}')

    #initialize model
    if args.ngpu > 0:
        cmd = utils.set_cuda_visible_device(args.ngpu)
        os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]

    model = gnn(args)
    print ('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, load_save_file=args.ckpt)

    #train and test dataset
    train_dataset = BaseDataset(train_keys, data_path, embedding_dim=args.embedding_dim)
    test_dataset = BaseDataset(test_keys, data_path, embedding_dim=args.embedding_dim)

    # num_train_chembl = len([0 for k in train_keys if 'iso' in k])
    # num_train_decoy = len([0 for k in train_keys if 'iso' not in k])
    # train_weights = [1/num_train_chembl if 'iso' in k else 1/num_train_decoy for k in train_keys]
    # train_sampler = UnderSampler(train_weights, len(train_weights), replacement=True)
                
    train_dataloader = DataLoader(train_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn) #, sampler = train_sampler)
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #loss function
    loss_fn = nn.BCELoss()

    # logging file
    log_file = open(os.path.join(log_dir, "%s_trace.txt"%args.dataset), "w", encoding="utf-8")
    log_file.write("epoch\ttrain_losses\ttest_losses\ttrain_roc\ttest_roc\ttime\n")

    for epoch in range(num_epochs):
        print("EPOCH", epoch)
        st = time.time()
        #collect losses of each iteration
        train_losses = [] 
        test_losses = [] 

        #collect true label of each iteration
        train_true = []
        test_true = []
        
        #collect predicted label of each iteration
        train_pred = []
        test_pred = []
        
        model.train()
        for sample in tqdm(train_dataloader):
            model.zero_grad()
            H, A1, A2, M, S, Y, V, _ = sample 
            H, A1, A2, M, S, Y, V = H.to(device), A1.to(device), A2.to(device),\
                                M.to(device), S.to(device), Y.to(device), V.to(device)
            
            #train neural network
            pred, attn_loss= model.train_model((H, A1, A2, V), (M, S))

            loss = loss_fn(pred, Y) + attn_loss
            loss.backward()
            optimizer.step()
            
            #collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().numpy())
            train_true.append(Y.data.cpu().numpy())
            train_pred.append(pred.data.cpu().numpy())
        
        model.eval()
        for sample in tqdm(test_dataloader):
            model.zero_grad()
            H, A1, A2, M, S, Y, V, _ = sample 
            H, A1, A2, M, S, Y, V = H.to(device), A1.to(device), A2.to(device),\
                                M.to(device), S.to(device), Y.to(device), V.to(device)
            
            #train neural network
            pred, attn_loss = model.train_model((H, A1, A2, V), (M, S))

            loss = loss_fn(pred, Y) + attn_loss
            
            #collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().numpy())
            test_true.append(Y.data.cpu().numpy())
            test_pred.append(pred.data.cpu().numpy())
            
        train_losses = np.mean(np.array(train_losses))
        test_losses = np.mean(np.array(test_losses))
        
        train_pred = np.concatenate(np.array(train_pred), 0)
        test_pred = np.concatenate(np.array(test_pred), 0)
        
        train_true = np.concatenate(np.array(train_true), 0)
        test_true = np.concatenate(np.array(test_true), 0)

        train_roc = roc_auc_score(train_true, train_pred) 
        test_roc = roc_auc_score(test_true, test_pred)
        
        end = time.time()
        print("%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" \
        %(epoch, train_losses, test_losses, train_roc, test_roc, end-st))

        log_file.write("%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" \
        %(epoch, train_losses, test_losses, train_roc, test_roc, end-st))

        name = save_dir + '/save_'+str(epoch)+'.pt'
        torch.save(model.state_dict(), name)

    log_file.close()

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print(s)

    main(args)
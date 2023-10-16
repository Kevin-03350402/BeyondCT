import json
import torch
from vit_pytorch.vit_3d_info import ViTInfo
from vit_pytorch.vit_3d import ViT

import torch.nn as nn
from utility import *
from dataloader import *
import sys
import os
import numpy as np
import json
import pandas as pd
import random
from torchsampler import ImbalancedDatasetSampler
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

import sklearn.utils






def load_model (args):
    if args.demos!=[]:
        if os.path.isfile(args.resume_path):
            print("Resumed: with demo")
            return torch.load(args.resume_path)
        else:
            print("Start new training: with demo")
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            print(device)
            vit_model_info = ViTInfo(
            image_size=args.image_size,
            frames=args.frames,
            image_patch_size=args.image_patch_size,
            frame_patch_size=args.frame_patch_size,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
            channels=args.channels, 
            info_size = len(args.demos),
            output_nums = len(args.lung_functions)
            )


            vit_model_info = vit_model_info.to(device)
            return vit_model_info
    else:
        if os.path.isfile(args.resume_path):
            print("Resumed: without demo")
            return torch.load(args.resume_path)
        else:
            print("Start new training: without demo")
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            print(device)
            vit_model_info = ViT(
            image_size=args.image_size,
            frames=args.frames,
            image_patch_size=args.image_patch_size,
            frame_patch_size=args.frame_patch_size,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
            channels=args.channels,
            output_nums = len(args.lung_functions)
            )


            vit_model_info = vit_model_info.to(device)
            return vit_model_info



def train_model (args,vit_model_info):
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

  
    train_df = pd.read_csv(args.train_data_path)
    valid_df = pd.read_csv(args.val_data_path)

    my_dataset = CustomDataset(train_df,use_aug= args.train_aug, bins = args.bins_width,
                               image_path_col_name = args.image_path_col_name, lung_functions = args.lung_functions, demos  = args.demos) # create your datset
    if args.data_balance:
        df_sampler = ImbalancedDatasetSampler(my_dataset)
    else:
        df_sampler = None
    my_dataloader = DataLoader(my_dataset,sampler=df_sampler,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True) # create your dataloader
    
    val_dataset = CustomDataset(valid_df,use_aug = args.val_aug, 
                                image_path_col_name = args.image_path_col_name, lung_functions = args.lung_functions, demos  = args.demos) # create your datset
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,sampler=None,num_workers=args.num_workers,pin_memory=True) # create your dataloader


    

    
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(vit_model_info.parameters(), lr=args.lr )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=args.factor, patience= args.patience, verbose=True, threshold=args.eps, threshold_mode="abs",min_lr = args.min_lr)
    
    best_train_mae = sys.maxsize
    best_val_mae = sys.maxsize
    best_train_lungfun = sys.maxsize
    best_val_lungfun = sys.maxsize

    
    for epoch in range(args.max_epochs):
        train_df = sklearn.utils.shuffle(train_df)

        running_loss = 0.0
        batch_count = 0
        epoch_acc = 0
        
        for i, data in enumerate(my_dataloader):
            vit_model_info.train()
            
            if (i > 0 and i % 1 == 0):
                print(f'train iteration: {i} epoch: {epoch} train L1 loss: {running_loss / batch_count:.5f} Current train percentage error: {epoch_acc/batch_count}')
            batch_count += 1
            demo, inputs, labels = data


            labels = labels[:,None]
            

            inputs = inputs.to(device)
            labels = labels.to(device)
            demo = demo.to(device)
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation

            if args.demos!=[]:
                outputs = vit_model_info(inputs,demo)
            else:
                outputs = vit_model_info(inputs)
            pred_res = outputs.flatten().tolist()
            true_label = labels.flatten().tolist()
            # record the percentage error
            epoch_acc += np.mean(np.abs((np.array(true_label) - np.array(pred_res))/np.array(pred_res)))
            outputs = outputs.to(torch.float64)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()

        # display statistics

        print(f'[epoch: {epoch + 1}] train L1 loss: {running_loss / batch_count:.5f}')
        with open("traing_log.txt", "a") as text_file:
            text_file.write(f'[epoch: {epoch + 1}] train L1 loss: {running_loss / batch_count:.5f}\n')
        best_train_mae = min(best_train_mae,running_loss/batch_count)
        best_train_lungfun = min(best_train_lungfun,epoch_acc/batch_count)
        val_loss = 0.0
        val_batch_count = 0
        epoch_acc_val = 0
        for i, data in enumerate(val_dataloader):
            vit_model_info.eval()
            if (i > 0 and i % 1 == 0):
                print(f'val iteration: {i} epoch: {epoch} val L1 loss: {val_loss / val_batch_count:.5f} Current train percentage error: {epoch_acc_val/val_batch_count}')
            demo, inputs, labels = data


            labels = labels[:,None]
            val_batch_count += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            demo = demo.to(device)
            # set optimizer to zero grad to remove previous epoch gradients
            # forward propagation
            if args.demos!=[]:
                outputs = vit_model_info(inputs,demo)
            else:
                outputs = vit_model_info(inputs)
            outputs = outputs.to(torch.float64)
            pred_res = outputs.flatten().tolist()
            true_label = labels.flatten().tolist()
            # record the percentage error
            epoch_acc_val += np.mean(np.abs((np.array(true_label) - np.array(pred_res))/np.array(pred_res)))
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        # display statistics

        best_val_mae = min(best_val_mae,val_loss/val_batch_count)
        print(f'[epoch: {epoch + 1}] validation L1 loss: {val_loss / val_batch_count:.5f}')
        with open("traing_log.txt", "a") as text_file:
            text_file.write(f'[epoch: {epoch + 1}] validation L1 loss: {val_loss / val_batch_count:.5f}\n')
        if epoch_acc_val/val_batch_count < best_val_lungfun:

            torch.save(vit_model_info, args.best_save_path)
        best_val_lungfun = min(best_val_lungfun,epoch_acc_val/val_batch_count)
        print(f'current train percentage error: {epoch_acc/batch_count} current val percentage error {epoch_acc_val/val_batch_count}')
        with open("traing_log.txt", "a") as text_file:
            text_file.write(f'current train percentage error: {epoch_acc/batch_count} current val percentage error {epoch_acc_val/val_batch_count}\n')
        
        print(f'best train MAE: {best_train_mae} best val MAE: {best_val_mae}')
        with open("traing_log.txt", "a") as text_file:
            text_file.write(f'best train MAE: {best_train_mae} best val MAE: {best_val_mae}\n')

        print(f'best train percentage error: {best_train_lungfun} best val percentage error: {best_val_lungfun}')
        with open("traing_log.txt", "a") as text_file:
            text_file.write(f'best train percentage error: {best_train_lungfun} best val percentage error: {best_val_lungfun}\n')
        scheduler.step(val_loss/val_batch_count)
        

        torch.save(vit_model_info, args.resume_path)
        torch.cuda.empty_cache()
        vit_model_info = vit_model_info.to(device)
    text_file.close()



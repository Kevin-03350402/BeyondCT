from torch.utils.data import TensorDataset, Dataset, DataLoader
from utility import *
import torch


class CustomDataset(Dataset):
    def __init__(self, dataframe,use_aug,image_path_col_name, lung_functions ,demos,bins = 100):
        self.df = dataframe
        

        
        self.bins = bins
        self.x = self.df.loc[:, image_path_col_name].values

        if len(lung_functions) == 1:
            y_index = lung_functions[0]
            self.y_all_label = self.df.loc[:,y_index].values
        else:
    
            self.y_all_label = self.df.loc[:, lung_functions].values

        
        
   
        self.demo = self.df.loc[:, demos].values
        
        self.len = self.x.shape[0]
        self.use_aug = use_aug

        
    def __len__(self):
        return self.len
    



    def __getitem__(self, idx):

        if self.use_aug:


            return (torch.Tensor(np.array(self.demo[idx])),torch.Tensor(np.array(nii2np(self.x[idx]))),self.y_all_label[idx])
        else:
            return (torch.Tensor(np.array(self.demo[idx])),torch.Tensor(np.array(nii2npval(self.x[idx]))),self.y_all_label[idx])
    def get_labels(self):

        return self.y_all_label//self.bins
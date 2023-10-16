from train import *
class Args:
    # Data configurations
    train_data_path = "train.csv"  # Path to the training dataset
    val_data_path = "validation.csv"  # Path to the val dataset
    batch_size = 32  # Batch size for training and validation
    train_aug = True
    val_aug = False
    # DataLoader configurations
    num_workers = 6  # Number of workers for data loading
    pin_memory = True  # Whether to use pinned memory for data loading
    data_balance = False # Set to true to add databalance, but will lead to slow convergence
    # Model configurations
    image_size = 256  # Image size for ViT
    frames = 256  # Number of frames for ViT
    image_patch_size = 4  # Image patch size for ViT
    frame_patch_size = 4  # Frame patch size for ViT
    dim = 128  # Dimension for ViT
    depth = 4  # Depth for ViT
    heads = 8  # Number of heads for ViT
    mlp_dim = 1024  # MLP dimension for ViT
    dropout = 0.1  # Dropout rate for ViT
    emb_dropout = 0.1  # Embedding dropout rate for ViT
    channels = 1  # Channels for ViT
    bins_width = 10
    # Training configurations
    lr = 0.0002  # Learning rate
    max_epochs = 1000  # Maximum number of training epochs
    factor = 0.5
    patience = 3
    eps = 0.01
    image_path_col_name = "image path"
    lung_functions = ["FEV1"]
    demos = [] # ['Age','Gender (female = 0)',	'Height(inch)',	'Weight(lbs)',	'Smoking_Status (0: current  1: former  2: else)',	'Cigs_Per_Day']
 
    min_lr = 1.0E-6

    best_save_path = "current_best.pt"  # Path to save the best model
    resume_path = "resume_model--.pt"  # Path template to save models every epoch

from torchsummary import summary

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    model = load_model(Args)
    # summary(model, input_size=(1, 256,256,256), batch_size = -1)
    train_model(Args, model)


    

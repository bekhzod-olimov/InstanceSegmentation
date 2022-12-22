from dataset import CustomDataset
from model import get_model
# from transformations import get_train_transforms, get_valid_transforms
from transformations import get_transform
from utils_train import train_one_epoch, evaluate
import torch, os, argparse, yaml
from torch.utils.data import DataLoader
import utils

def run(args):
    
    root = args.root
    bs = args.batch_size
    device = args.device
    lr = args.learning_rate
    epochs = 50
    num_classes = 2
    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    trans_tr, trans_val = get_transform(train=True), get_transform(train=True)
    tr_ds = CustomDataset(root, trans_tr)
    val_ds = CustomDataset(root, trans_val)
    print(f"There are {len(tr_ds)} number of images in the dataset!")
    
    tr_dl = DataLoader(tr_ds, batch_size=int(bs/2), shuffle=True, collate_fn=utils.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=int(bs/4), shuffle=False, collate_fn=utils.collate_fn)   
    
    model = get_model()    
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    
    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, opt, tr_dl, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_dl, device)    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Instance Segmentation Training Arguments')
    parser.add_argument("-r", "--root", type=str, default='data', help="Path to the data")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:0', help="GPU device number")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-3, help="Learning rate value") # from find_lr
    args = parser.parse_args() 
    
    run(args) 
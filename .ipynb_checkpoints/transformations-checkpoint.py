from torchvision import transforms
import torch
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# def get_train_transforms():
    
#     return transforms.Compose([
#            transforms.ToTensor(),
#            transforms.ConvertImageDtype(torch.float),
#            transforms.RandomHorizontalFlip(p=0.5),
#            ])


# def get_valid_transforms():
    
#     return transforms.Compose([
#            transforms.ToTensor(),
#            transforms.ConvertImageDtype(torch.float),
#            ])
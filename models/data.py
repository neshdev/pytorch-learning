import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms


def normalization_factors():
    dataset = MNIST('/data',train=True)
    train = dataset.train_data.float()
    n, _, _ = train.shape
    train_mean = train.mean(dim=0)
    train_std = train.std(dim=0) + 1e-6
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(train_mean,train_std),
    ])
    return transform
    

def train_dataloader(config)-> DataLoader:
    normal_transform = normalization_factors()
    loader = DataLoader(MNIST('/data',train=True,transform=normal_transform),batch_size=config.batch_size,shuffle=True)
    
    return loader,transform

def test_dataloader(config)-> DataLoader:
    normal_transform = normalization_factors()
    dataset = MNIST('/data',train=False,transform=normal_transform)
    n = len(dataset.test_data)
    print(f'test batch size { config.get("batch_size",n) }')
    loader = DataLoader(dataset,batch_size=config.get("batch_size",n))    
    return loader
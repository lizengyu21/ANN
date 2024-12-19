import jittor as jt
import logging
from PIL import Image
import os

from .dataset import CUB

from jittor import transform
from jittor.dataset import RandomSampler, SequentialSampler
from jittor.dataset import DataLoader


logger = logging.getLogger(__name__)

def get_loader(args):
    if args.dataset == 'CUB_200_2011':
        train_transform = transform.Compose([
            transform.Resize((600, 600)),# 默认是BILINEAR
            transform.RandomCrop((args.img_size, args.img_size)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transform.Compose([
            transform.Resize((600, 600)),# 默认是BILINEAR
            transform.RandomCrop((args.img_size, args.img_size)),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = CUB(root=args.data_root, data_len=64, train_transform=train_transform, test_transform=test_transform)
        trainset = dataset.get_train_dataset()
        testset = dataset.get_test_dataset()
    else:
        raise ValueError('Invalid dataset name: {}'.format(args.dataset))
    
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, num_workers=4, drop_last=True, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=args.eval_batch_size, num_workers=4, sampler=test_sampler)
    
    return train_loader, test_loader
    
    
    
    
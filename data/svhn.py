import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args


class SVHN:
    def __init__(self, args):
        super(SVHN, self).__init__()

        data_root = os.path.join(args.data, "svhn")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        trans = transforms.Compose(
            [
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        trans_val = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train_dataset = torchvision.datasets.SVHN(
            root=data_root,
            split='train',
            download=True,
            transform=trans,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        test_dataset = torchvision.datasets.SVHN(
            root=data_root,
            split='test',
            download=True,
            transform=trans_val,
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )

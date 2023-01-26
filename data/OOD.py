import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args

class OOD_Loader(object):
    def __init__(self, args, data, trans_val=None):
        super(OOD_Loader, self).__init__()

        data_root = os.path.join(args.data, data)

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
        if trans_val is None:
            trans_val = transforms.Compose([
                    transforms.ToTensor(),
                ]
            )

        dataset = torchvision.datasets.ImageFolder(data_root, trans_val)
        self.val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs
        )

class SUN(OOD_Loader):
    def __init__(self, args):
        trans_val = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        super(SUN, self).__init__(args, "SUN", trans_val)

class Places(OOD_Loader):
    def __init__(self, args):
        trans_val = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        super(Places, self).__init__(args, "Places", trans_val)

class dtd(OOD_Loader):
    def __init__(self, args):
        trans_val = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        super(dtd, self).__init__(args, "dtd", trans_val)

class iNaturalist(OOD_Loader):
    def __init__(self, args):
        trans_val = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        super(iNaturalist, self).__init__(args, "iNaturalist", trans_val)

class LSUN(OOD_Loader):
    def __init__(self, args):
        super(LSUN, self).__init__(args, "LSUN")

class iSUN(OOD_Loader):
    def __init__(self, args):
        super(iSUN, self).__init__(args, "iSUN")

class LSUN_resize(OOD_Loader):
    def __init__(self, args):
        super(LSUN_resize, self).__init__(args, "LSUN_resize")

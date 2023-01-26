import numpy as np
import torch
import os
import pickle

from torchvision import datasets, transforms
from PIL import Image, ImageOps 
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class imagenet(torch.utils.data.Dataset):

    def __init__(self, transform=None, data_root=None, img_size=64):
        self.transform = transform
        self.S = np.zeros(11, dtype=np.int32)
        self.img_size = img_size
        self.labels = []
        self.data_root = data_root
        for idx in range(1, 11):
            data_file = os.path.join(self.data_root, 'train_data_batch_{}'.format(idx))
            d = unpickle(data_file)
            y = d['labels']
            y = [i-1 for i in y]
            self.labels.extend(y)
            self.S[idx] = self.S[idx-1] + len(y)

        self.labels = np.array(self.labels)
        self.N = len(self.labels)
        self.curr_batch = -1

        self.offset = 0     # offset index

    def load_image_batch(self, batch_index):
        data_file = os.path.join(self.data_root, 'train_data_batch_{}'.format(batch_index))
        d = unpickle(data_file)
        x = d['data']
        
        img_size = self.img_size
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))

        self.batch_images = x
        self.curr_batch = batch_index

    def get_batch_index(self, index):
        j = 1
        while index >= self.S[j]:
            j += 1
        return j

    def load_image(self, index):
        batch_index = self.get_batch_index(index)
        if self.curr_batch != batch_index:
            self.load_image_batch(batch_index)
        
        return self.batch_images[index-self.S[batch_index-1]]

    def __getitem__(self, index):
        index = (index + self.offset) % self.N

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index] 

    def __len__(self):
        return self.N

class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()
        data_root = os.path.join(args.data, "ImageNet")
        use_cuda = torch.cuda.is_available()
        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.ToPILImage(), 
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor()
        ])

        train_dataset = imagenet(trans, data_root)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.ood_batch_size, shuffle=False, **kwargs
        )
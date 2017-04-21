
import os
import argparse
import json
import multiprocessing

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image


from asg.datadirectory import data_directory
from asg.model import Net
from asg.logger import Logger


# find ./images_full -mindepth 2 -type f -exec mv -t ./images_full -i '{}' +

# Argument settings
parser = argparse.ArgumentParser(
    description='PyTorch Automated Storyboard Generator - Cache Image Embeddings')

parser.add_argument('--model', type=str,
                    help='Path to model definition')
parser.add_argument('--output', type=str,
                    help='Path to output data')
parser.add_argument('--seed', type=int, default=451,
                    help='Random seed. Default=451')
parser.add_argument('--batch', type=int, default=32,
                    help='Batch size. Default=32')
parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                    help='Number of workers for data loader. Defaults to system cores')

opt = parser.parse_args()
# opt = parser.parse_args(([
#     '--path',
#     'models/model_a532783_epoch_0000018.pth'
# ]))

Logger.log("Init")


def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageSimple(torch.utils.data.Dataset):

    def __init__(self,
                 group,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self._image_path = os.path.join(data_directory, group)

        self._image_paths = [image_path for image_path in os.listdir(self._image_path)]


    def __getitem__(self, index):

        filename = self._image_paths[index]
        path = os.path.join(self._image_path, filename)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return filename, img

    def __len__(self):
        return len(self._image_paths)

loader_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_loader_train = ImageSimple('train',
                                 transform=loader_transforms)

loader_train = torch.utils.data.DataLoader(image_loader_train,
                               batch_size=opt.batch, shuffle=False,
                               num_workers=opt.workers, pin_memory=True)


Logger.log("Starting with model for {}".format(opt.model))

Logger.log("Loading Network")
CUDA_AVAILABLE = torch.cuda.is_available()

torch.manual_seed(opt.seed)
if CUDA_AVAILABLE:
    torch.cuda.set_device(1)
    torch.cuda.manual_seed(opt.seed)


def variable(target):
    """Convert tensor to a variable"""
    if CUDA_AVAILABLE:
        target = target.cuda()
    return Variable(target)

net = Net()
if CUDA_AVAILABLE:
    net = net.cuda()

net.load_state_dict(torch.load(opt.model))


def text_size_to_variables(text_sizes):
    return [variable(torch.LongTensor([text_size])) for text_size in text_sizes]

net.eval()
text = variable(torch.randn(opt.batch, 2, 300))
text_sizes_var = text_size_to_variables([1] * opt.batch)

filename_to_embedding = {}
for index, (filenames, images) in enumerate(loader_train):

    images = variable(images)
    output_image_var, _ = net(images, text, text_sizes_var)

    data = output_image_var.data.cpu()
    for idx in range(opt.batch):
        filename = filenames[idx]
        tensor = data[idx].tolist()
        filename_to_embedding[filename] = tensor

    if index % (len(loader_train) // 50) == 0:
        Logger.log('Completed {:0.2f}% {:0>6}/{:0>6}'.format(
                100 * index / len(loader_train),
                index,
                len(loader_train)
            ))
    if index > 50:
        break


Logger.log("Reading Complete")

with open(opt.output, 'w') as fp:
    json.dump(filename_to_embedding, fp)

Logger.log("Writing Complete")

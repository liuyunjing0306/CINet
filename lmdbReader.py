import torch
import io
import os

import lmdb
import six
import sys
import random
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset


# # 24.10.4 来源于lmdbReader2，将其中的打印和检测输出删除了
## 文件夹名称关联 标签
class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, reverse=False, alphabet=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error index: %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            label = strQ2B(label)
            # label += '$'
            label = label.lower()
            if self.transform is not None:
                img = self.transform(img)
        return (img, label, index)


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


class resizeNormalize(object):

    def __init__(self, size, test=False, interpolation=Image.BILINEAR):
        self.test = test
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        width, height = img.size
        if 1.5 * width < height:
            img = img.transpose(Image.ROTATE_90)
        img = img.resize(self.size, self.interpolation)

        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

# if __name__ == '__main__':
#     dataset_root = r'D:\LYJ_Project\Project2\lmdb\train_3755'
#     dataset = lmdbDataset(dataset_root, resizeNormalize((128, 128), test=False))
#     train_dataset = []
#     train_dataset.append(dataset)
#     train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset_total, batch_size=31920, shuffle=True, num_workers=0)  ## 测试 一轮全部读出来，31920
#     train_loader = train_dataloader
#     dataloader = iter(train_loader)
#     data = next(dataloader)
#     # image, label = data

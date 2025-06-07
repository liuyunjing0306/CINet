# -*- coding: utf-8 -*-
import argparse
import glob
import io
import os
import pathlib
import threading

import cv2 as cv
import lmdb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

## 文件夹名称关联 标签
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

root_path = pathlib.Path('D:\\LYJ_Project\\Project_Differ_Mask\\Mask_0.4\\')

output_path = os.path.join(root_path, pathlib.Path('lmdb_test1'))  # 输出lmdb的文件夹名称lmdb_test/lmdb/lmdb_test2
train_path = os.path.join(root_path, pathlib.Path('ProductImage\\original_data\\train_mask'))  # 需要处理训练的数据集的路径
val_path = os.path.join(root_path, pathlib.Path('ProductImage\\original_data\\test1_mask'))  # 需要处理测试的数据集的路径

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, bytes):
                # 图片类型为bytes
                txn.put(k.encode(), v)
            else:
                # 标签类型为str, 转为bytes
                txn.put(k.encode(), v.encode())  # 编码


def show_image(samples):
    plt.figure(figsize=(20, 10))
    for pos, sample in enumerate(samples):
        plt.subplot(4, 5, pos + 1)
        plt.imshow(sample[0])
        # plt.title(sample[1])
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
    plt.show()


def lmdb_test(root):
    env = lmdb.open(
        root,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)

    if not env:
        print('cannot open lmdb from %s' % root)
        return

    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))

    with env.begin(write=False) as txn:
        samples = []
        for index in range(1, n_samples + 1):
            img_key = 'image-%09d' % index
            img_buf = txn.get(img_key.encode())
            buf = io.BytesIO()
            buf.write(img_buf)
            buf.seek(0)
            try:
                img = Image.open(buf)
            except IOError:
                print('Corrupted image for %d' % index)
                return
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            print(n_samples, len(img.split()), label)
            samples.append([img, label])
            if index == 5:
                # show_image(samples)
                # samples = []
                break


def lmdb_init(directory, out):
    # 读取文件夹创建字典
    all_image_path = []
    all_image_label = []
    all_image_key = []  # .mdb数据库文件保存了两种数据，一种是图片数据，一种是标签数据，它们各有其key
    all_label_key = []

    # 遍历 directory 下的所有子文件夹
    all_cnt = 0
    dir_cot = 0
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        print(str(dir_cot) + ':' + folder_path)
        dir_cot += 1

        # if dir_cot > 2:
        #     continue

        # 获取当前文件夹中所有的图片
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, file))]
        # 遍历当前文件夹中所有图片
        for file in files:
            all_cnt += 1
            file_name = os.path.basename(file)
            file_name_without_ext = os.path.splitext(file_name)[0]
            label = folder_name + '_' + file_name_without_ext
            image_key = 'image-%09d' % all_cnt
            label_key = 'label-%09d' % all_cnt
            all_image_path.append(file)
            all_image_label.append(label)
            all_image_key.append(image_key)
            all_label_key.append(label_key)

    pbar = tqdm(all_image_path)

    # 计算所需内存空间
    image_cnt = len(all_image_path)
    data_size_per_img = cv.imdecode(np.fromfile(all_image_path[0], dtype=np.uint8), cv.IMREAD_UNCHANGED).nbytes
    # 一个类中所有图片的字节数
    data_size = data_size_per_img * image_cnt * 2
    # 创建lmdb文件
    if not os.path.exists(out):
        os.makedirs(out)
    env = lmdb.open(out, map_size=data_size)

    cnt = 0
    # for file_name in pbar:
    for file_path in pbar:
        # 读取图片路径和对应的标签
        image = file_path
        label = all_image_label[cnt]
        image_key = all_image_key[cnt]
        label_key = all_label_key[cnt]
        cnt += 1

        cache = {}
        with open(image, 'rb') as fs:
            image_bin = fs.read()

        cache[image_key] = image_bin
        cache[label_key] = label.encode()

        if len(cache) != 0:
            write_cache(env, cache)

        pbar.set_description(
            f'character[{label} | nSamples: {cnt} |')

    write_cache(env, {'num-samples': str(cnt)})
    env.close()


def begin(mode, left, right, valid=False):
    if mode == 'train':
        path = os.path.join(output_path, pathlib.Path(mode + '_' + str(right)))
        if not valid:
            lmdb_init(train_path, path)
        else:
            print(f"show:{valid},path:{path}")
            lmdb_test(path)
    elif mode == 'test':
        path = os.path.join(output_path, pathlib.Path(mode + '_' + str(right - left)))
        if not valid:
            lmdb_init(val_path, path)
        else:
            print(f"show:{valid},path:{path}")
            lmdb_test(path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("train", action="store_true", help="generate train lmdb")
    parser.add_argument("--test", action="store_true", help="generate test lmdb")
    parser.add_argument("--all", action="store_true", help="generate all lmdb")
    parser.add_argument("--show", action="store_true", help="show result")
    parser.add_argument("--start", type=int, default=0, help="class start from where,default 0")
    parser.add_argument("--end", type=int, default=1126,
                        help="class end from where,default 3755")  ## 测试(1126 data(data2)/3755test_all2)和train不一样，train：3755

    args = parser.parse_args()

    train = args.train
    test = args.test
    build_all = args.all
    start = args.start
    end = args.end
    show = args.show

    if train:
        print(f"args: mode=train, [start:end)=[{start}:{end})")
        begin(mode='test', left=start, right=end, valid=show)  # 控制是处理是test还是train数据集

        # 内容_编号_风格编号

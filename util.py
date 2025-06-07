import torch
import os
import shutil
import pickle as pkl
import torch.nn as nn
import copy

from lmdbReader import lmdbDataset, resizeNormalize
from config import config
from shutil import copyfile
import re

mse_loss = nn.MSELoss()
alphabet_character_file = open(config['alpha_path'], 'r', encoding='UTF-8')
alphabet_character = list(alphabet_character_file.read().strip())
alphabet_character_raw = ['START']

for item in alphabet_character:
    alphabet_character_raw.append(item)

alphabet_character_raw.append('END')
alphabet_character = alphabet_character_raw  # alphabet_character list:3757 即 ['START', '啊',......'END']

alp2num_character = {}

for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index  # alp2num_character为字典dict:3757 即{'START':0,'啊':1,.......'END': 3756}


def get_dataloader(root, shuffle=False):
    if root.endswith('pkl'):
        f = open(root, 'rb')
        dataset = pkl.load(f)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=8,
        )
    else:
        dataset = lmdbDataset(root, resizeNormalize((config['imageW'], config['imageH'])))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=8,
        )
    return dataloader, dataset


def get_data_package():
    # train_dataset = []
    # for dataset_root in config['train_dataset'].split(','):
    #     _, dataset = get_dataloader(dataset_root, shuffle=True)
    #     train_dataset.append(dataset)
    # train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    #
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=0,
    # )

    test_dataset = []
    for dataset_root in config['test_dataset'].split(','):
        _, dataset = get_dataloader(dataset_root, shuffle=True)
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=0,
    )

    # return train_dataloader, test_dataloader
    return test_dataloader


# 此部分类似于CLIP
r2num = {}
alphabet_radical = []
alphabet_radical.append('PAD')
lines = open(config['radical_path'], 'r', encoding='utf-8').readlines()
for line in lines:
    alphabet_radical.append(line.strip('\n'))
alphabet_radical.append('$')
for index, char in enumerate(alphabet_radical):
    r2num[char] = index

dict_file = open(config['decompose_path'], 'r', encoding='utf-8').readlines()
char_radical_dict = {}
for line in dict_file:
    line = line.strip('\n')
    try:
        char, r_s = line.split(':')
    except:
        char, r_s = ':', ':'
    if 'rsst' not in config['decompose_path']:
        char_radical_dict[char] = r_s.split(' ')
    else:
        char_radical_dict[char] = list(''.join(r_s.split(' ')))
    # char, r_s = line.split(':')
    # char_radical_dict[char] = r_s.split(' ')


def convert_char(label):
    r_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        r_tmp.append('$')
        r_label.append(r_tmp)

    text_tensor = torch.zeros(batch, 30).long().cuda()
    for i in range(batch):
        tmp = r_label[i]
        for j in range(len(tmp)):
            text_tensor[i][j] = r2num[tmp[j]]
    return text_tensor


def get_radical_alphabet():
    return alphabet_radical


# def converter(label):
#     string_label = label
#     label = [i for i in label]
#     alp2num = alp2num_character
#
#     batch = len(label)
#     length = torch.Tensor([len(i) for i in label]).long().cuda()
#     max_length = max(length)
#
#     text_input = torch.zeros(batch, max_length).long().cuda()
#     for i in range(batch):
#         temp = len(label[i])
#         for j in range(len(label[i]) - 1):
#             text_input[i][j + 1] = alp2num[label[i][j]]
#
#     sum_length = sum(length)
#     text_all = torch.zeros(sum_length).long().cuda()
#     start = 0
#     for i in range(batch):
#         for j in range(len(label[i])):
#             if j == (len(label[i]) - 1):
#                 text_all[start + j] = alp2num['END']
#             else:
#                 text_all[start + j] = alp2num[label[i][j]]
#         start += len(label[i])
#
#     return length, text_input, text_all, string_label
# 修改过的converter
def converter(label):
    if isinstance(label, str):
        label = list(label)  # 将字符串转换为列表

    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character  # alp2num为字典dict:3757 即{'START':0,'啊':1,.......'END': 3756}

    batch = len(label)
    length = torch.Tensor([int(len(i)) for i in label]).long().cuda()
    max_length = max(length)

    # 初始化text_input
    text_input = torch.zeros(batch, max_length + 1).long().cuda()

    # 填充text_input
    for i in range(batch):
        for j in range(len(label[i])):  # 执行len(label[i])次
            text_input[i][j + 1] = alp2num[label[i][j]]  #'我$'label[0][0]表示我,label[0][1]表示$

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()

    start = 0
    # 填充text_all
    for i in range(batch):
        for j in range(len(label[i])):
            text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])

    return length, text_input, text_all, string_label


def get_alphabet():
    return alphabet_character


def tensor2str(tensor):
    alphabet = get_alphabet()
    string = ""
    for i in tensor:
        if i == (len(alphabet) - 1):
            continue
        string += alphabet[i]
    return string


def must_in_screen():
    text = os.popen('echo $STY').readlines()
    string = ''
    for line in text:
        string += line
    if len(string.strip()) == 0:
        print("run in the screen!")
        exit(0)


def saver():
    try:
        shutil.rmtree('./history/{}'.format(config['exp_name']))
    except:
        pass
    # os.mkdir('./history/{}'.format(config['exp_name'])) # 原始
    os.makedirs('./history/{}'.format(config['exp_name']), exist_ok=True)


if __name__ == '__main__':
    length, text_input, text_all, string_label = converter(['啊', '你', '他'])
    print(length)
    print(text_input)
    print(text_all)
    print(string_label)

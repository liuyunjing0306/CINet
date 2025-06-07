# 使用的是lmdb数据
import torch.nn as nn
import torch
from model.Recognition_Net import Transformer
from model.Inpainting_Net import *
from util import convert_char, get_alphabet, get_radical_alphabet, converter, tensor2str, get_data_package

from config import config
from model.clip import CLIP
import os
import six
from PIL import Image
from lmdbReader import resizeNormalize
# Batch_Tensor_to_Image2 使用的是torchvision.utils 保存
from Batch_Tensor_to_Image2 import restore_image
# Batch_Tensor_to_Image使用的是PIL保存图像
# from Batch_Tensor_to_Image import restore_image
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)

test_loader = get_data_package()
alphabet = get_alphabet()
radical_alphabet = get_radical_alphabet()


def get_test_part_label(test_labels):
    test_parts_label = []
    for label in test_labels:
        char = label.split("_")[0]
        test_parts_label.append(char)
    return test_parts_label


def get_images_and_style_index(labels):
    images = []
    styles_indexs = []  # 风格的标签
    parts_label = []  # 只有内容的标签
    root_path = config['test_clean_path']
    for label in labels:
        char = label.split("_")[0]
        name = label.split("_")[1]
        style_index = label.split("_")[2]

        styles_indexs.append(style_index)
        parts_label.append(char)
        image_path = os.path.join(root_path, char, f"{name}_{style_index}.png")  # 将jpg修改成png

        with open(image_path, 'rb') as f:
            imgbuf = f.read()

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        try:
            img = Image.open(buf).convert('RGB')
        except IOError:
            print('Corrupted image at %s' % image_path)
            return None

        transform = resizeNormalize((config['imageW'], config['imageH']))
        if transform is not None:
            img = transform(img)
            images.append(img)

    images_tensor = torch.stack(images).cuda()

    return images_tensor, styles_indexs, parts_label


def test():
    model_recog = Transformer().cuda()
    model_recog = nn.DataParallel(model_recog)
    # 修复分支
    model_inpaint = InpaintingNet(gpu=0).cuda()
    model_inpaint = nn.DataParallel(model_inpaint)
    torch.cuda.empty_cache()
    model_recog_path = "/RecogNet_state_200.pt"
    model_inpaint_path = "/InpaintNet_state_200.pt"
    model_recog.eval()
    model_inpaint.eval()
    try:
        checkpoint1 = torch.load(model_recog_path, weights_only=True)
        model_recog.load_state_dict(checkpoint1, strict=False)
        print("模型加载成功！")
    except RuntimeError as e:
        print("模型加载失败:", e)
        return

    try:
        checkpoint2 = torch.load(model_inpaint_path, weights_only=True)
        model_inpaint.load_state_dict(checkpoint2, strict=False)
        print("模型加载成功！")
    except RuntimeError as e:
        print("模型加载失败:", e)
        return

    print("Start Eval!")
    # 查看eval是否固定住网络
    # for name, module in model_inpaint.named_modules():
    #     print(name, module.training)
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('test:', test_loader_len)

    with torch.no_grad():
        # 此处数据处理使用的lmdb的数据形式
        for iteration in range(test_loader_len):
            data = next(dataloader)
            image, label, _ = data
            image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
            # 添加 转换label
            test_part_label = get_test_part_label(label)
            image_clean, styles_labels, parts_label = get_images_and_style_index(label)
            # if test_part_label == parts_label:
            #     print("两个标签一样")
            # 将styles_labels list形式转成成int
            styles_labels_tensor = torch.tensor(list(map(int, styles_labels))).cuda()
            # length, text_input, text_gt, string_label = converter(test_part_label)
            # max_length = max(length)
            max_length = 1
            batch = image.shape[0]
            pred = torch.zeros(batch, 1).long().cuda()
            image_features = None
            # prob = torch.zeros(batch, max_length).float()

            for i in range(max_length):
                length_tmp = torch.zeros(batch).long().cuda() + i + 1
                result = model_recog(image, length_tmp, pred, conv_feature=image_features)
                prediction = result['pred']
                prediction = prediction / prediction.norm(dim=1, keepdim=True)
                # 修复分支损失和风格损失

                image_conv = result['inpaint_map']
                text_conv = prediction

                latent = model_inpaint(image_conv, text_conv)
                latent_style = model_inpaint.module.getLatent(latent)

                style_output, dot_output = model_inpaint.module.getSim(latent_style)

                # 为每个样本选择相应的风格特征
                style_emb = get_cuda(
                    style_output.clone()[torch.arange(style_output.size(0)), styles_labels_tensor.squeeze().long()],
                    0)  # (128,1024)
                style_emb = style_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16,
                                                                         16)  # 扩维(128,1024,8,8)  # 修改成（-1，-1，16，16）
                add_latent = latent + style_emb
                encoder_out1 = result['encoder_out1']
                encoder_out2 = result['encoder_out2']
                encoder_out3 = result['encoder_out3']
                encoder_out4 = result['encoder_out4']
                inpaint_output = model_inpaint.module.getOutput(encoder_out1, encoder_out2, encoder_out3, encoder_out4,
                                                                add_latent)  # inpaint_output['rec_image4']是8倍插值
                inpaint_output_1x = inpaint_output['rec_image1']  # 未插
                # inpaint_output_2x = inpaint_output['rec_image2']
                # inpaint_output_4x = inpaint_output['rec_image3']
                # inpaint_output_8x = inpaint_output['rec_image4']
                # 添加将网络的输出转化成图像的代码
                # 恢复图像格式
                restore_image(inpaint_output_1x, label)


if __name__ == '__main__':
    test()

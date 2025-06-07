import torch.nn as nn
import numpy as np
from model.Inpainting_Net import *
from model.Style_Classify import Classifier
from sklearn.metrics import accuracy_score
import datetime
import torch.optim as optim
from util import convert_char, get_alphabet, get_radical_alphabet, converter, tensor2str, get_data_package
from config import config
from torch.utils.tensorboard import SummaryWriter
from model.Recognition_Net import Transformer
from model.clip import CLIP
import os
import six
from PIL import Image
from lmdbReader import resizeNormalize
import random

writer = SummaryWriter('runs/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

alphabet = get_alphabet()  # 此部分CLIP没有
radical_alphabet = get_radical_alphabet()  # 此部分类似CLIP
# print('alphabet', alphabet)
# 识别分支
model_recog = Transformer().cuda()
model_recog = nn.DataParallel(model_recog)
# 修复分支
model_inpaint = InpaintingNet(gpu=0).cuda()
model_inpaint = nn.DataParallel(model_inpaint)

# 风格网络的初始化
dis_model = get_cuda(Classifier(gpu=0), 0)
dis_model = nn.DataParallel(dis_model)
if config['resume'].strip() != '':
    model_recog.load_state_dict(torch.load(config['resume']))
    print('loading！！！')
# 识别网络的优化器和学习率调制器
model_recog_optimizer = optim.Adadelta(model_recog.parameters(), lr=config['lr'], rho=0.9, weight_decay=1e-4)
model_recog_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_recog_optimizer, T_0=10, T_mult=1)

model_inpaint_optimizer = optim.Adam(model_inpaint.parameters(), lr=0.0002)
model_inpaint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_inpaint_optimizer, T_max=50)

# 风格网络优化器和学习率调制器
dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=0.0001)
dis_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer, T_max=50)  #
# 识别网络的损失函数定义
criterion = torch.nn.CrossEntropyLoss().cuda()
criterion_dis = torch.nn.MSELoss().cuda()
# 修复网络的损失函数定义
criterion_pixelwise = torch.nn.L1Loss().cuda()
# 风格网络的损失函数定义
dis_criterion = torch.nn.CrossEntropyLoss().cuda()

best_acc = -1

# times = 0

# train_loader, test_loader = get_data_package()
test_loader = get_data_package()

clip_model = CLIP(embed_dim=2048, context_length=30, vocab_size=len(radical_alphabet), transformer_width=512,
                  transformer_heads=8, transformer_layers=12).cuda()
clip_model = nn.DataParallel(clip_model)

try:
    checkpoint = torch.load(config['radical_model'], weights_only=True)
    clip_model.load_state_dict(checkpoint)
    print("模型加载成功！")
except RuntimeError as e:
    print("模型加载失败:", e)

char_file = open(config['alpha_path'], 'r', encoding='UTF-8').read()
chars = list(char_file)
tmp_text = convert_char(chars)  # 将3755个汉字拆分的索引分批次(/100)送到encode_text中取到预训练的文本特征
# selected_vector2=tmp_text[1946] 内监视
text_features = []
iters = len(chars) // 100
text_features.append(
    torch.zeros([1, 2048]).cuda())  # text_features添加2个向量的原因是为了能正确的索引，converter处理的 alp2num = alp2num_character为3757
with torch.no_grad():
    for i in range(iters + 1):
        s = i * 100
        e = (i + 1) * 100
        if e > len(chars):
            e = len(chars)
        text_features_tmp = clip_model.module.encode_text(tmp_text[s:e])
        text_features.append(text_features_tmp)
    text_features.append(torch.ones([1, 2048]).cuda())
    text_features = torch.cat(text_features, dim=0).detach()


# 在3755上添加了2个向量，最前面一个，最后面一个. char是3755，第一个啊为0，但是alphabet为3757，啊为1


# 根据mask图像的label 读取本地干净图像和字体风格并处理  源get_images(修改而来)
def get_images_and_style_index(labels):
    images = []
    styles_indexs = []  # 风格的标签
    parts_label = []  # 只有内容的标签
    root_path = config['train_clean_path']
    for label in labels:
        char = label.split("_")[0]
        name = label.split("_")[1]
        style_index = label.split("_")[2]

        styles_indexs.append(style_index)
        parts_label.append(char)
        image_path = os.path.join(root_path, char, f"{name}_{style_index}.jpg")

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


#  从本地获取所有的label 并分组，要求每一组中的内容不一样，把分完的组返回
def get_all_label():
    root_path = config['train_dataset_mask_image']
    batch_size = int(config['batch'])

    grouped_labels = []
    temp_label = []
    used_chars = set()

    # 读取所有图片名称
    all_labels = []
    for dirpath, _, filenames in os.walk(root_path):
        folder_name = os.path.basename(dirpath)  # 获取父文件夹名称
        for filename in filenames:
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # 支持的图片格式
                image_name = os.path.splitext(filename)[0]  # 去掉后缀，得到图片名称
                label = f"{folder_name}_{image_name}"  # 生成标签，格式为 父文件夹名_图片名
                all_labels.append(label)

    # 打乱所有标签的顺序
    random.shuffle(all_labels)

    reserved_labels = []

    # 分组处理标签
    for label in all_labels:
        char = label.split("_")[0]  # 获取标签的第一个字，即父文件夹名称

        if char not in used_chars:
            used_chars.add(char)
            temp_label.append(label)

            if len(temp_label) == batch_size:
                random.shuffle(temp_label)  # 打乱当前组内标签顺序
                grouped_labels.append(temp_label)

                # 重置临时列表和字符集
                temp_label = []
                used_chars = set()
        else:
            reserved_labels.append(label)

    # 处理保留的标签
    while len(reserved_labels) > batch_size:
        remove_label = []

        for label in reserved_labels:
            char = label.split("_")[0]

            if char not in used_chars:
                used_chars.add(char)
                temp_label.append(label)
                remove_label.append(label)

                if len(temp_label) == batch_size:
                    random.shuffle(temp_label)  # 打乱当前组内标签顺序
                    grouped_labels.append(temp_label)

                    # 重置临时列表和字符集
                    temp_label = []
                    used_chars = set()

        for label in remove_label:
            reserved_labels.remove(label)

    # 在返回之前检查每一组的标签是否符合要求
    if check_labels_group(grouped_labels):
        print("检查通过，每组中的第一个字都不一样。")
    else:
        print("检查失败，有组内存在重复的第一个字。")

    return grouped_labels


def check_labels_group(grouped_labels):
    """
    检查每一组中分割后前一个字是否不重复
    """
    for group in grouped_labels:
        first_chars = [label.split("_")[0] for label in group]
        if len(first_chars) != len(set(first_chars)):
            return False  # 如果有重复的第一个字，返回 False
    return True  # 如果所有组都通过检查，返回 True


# 通过labels从本地获取mask对应的图片
def get_images(labels):
    images = []

    root_path = config['train_dataset_mask_image']
    for label in labels:
        char = label.split("_")[0]
        name = label.split("_")[1]
        style = label.split("_")[2]
        image_path = os.path.join(root_path, char, f"{name}_{style}.jpg")

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

    images_tensor = torch.stack(images)

    return images_tensor


##
def train(epoch, iteration, image, length, text_input, text_gt, image_clean, styles_labels_tensor, iter_len, acc_style,
          times):
    # global times

    model_recog.train()
    model_recog_optimizer.zero_grad()

    model_inpaint.train()
    model_inpaint_optimizer.zero_grad()

    dis_model.train()
    dis_optimizer.zero_grad()

    reg_list = []
    for item in text_gt:
        reg_list.append(text_features[item].unsqueeze(0))
    reg = torch.cat(reg_list, dim=0)

    # 识别分支损失
    result = model_recog(image, length, text_input)
    text_pred = result['pred']
    text_pred = text_pred / text_pred.norm(dim=1, keepdim=True)
    final_res = text_pred @ text_features.t()

    loss_rec = criterion(final_res, text_gt)
    loss_dis = - criterion_dis(text_pred, reg)
    loss_text = loss_rec + 0.001 * loss_dis

    # 修复分支损失和风格损失
    image_conv = result['inpaint_map']
    text_conv = text_pred
    latent = model_inpaint(image_conv, text_conv)
    latent_style = model_inpaint.module.getLatent(latent)
    style_output, dot_output = model_inpaint.module.getSim(latent_style)
    # temp2 = styles_labels_tensor.squeeze().long() # 测试
    # 为每个样本选择相应的风格特征
    style_emb = get_cuda(
        style_output.clone()[torch.arange(style_output.size(0)), styles_labels_tensor.squeeze().long()],
        0)  # (128,1024)
    style_emb = style_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)  # 扩维(128,1024,8,8)
    add_latent = latent + style_emb
    encoder_out1 = result['encoder_out1']
    encoder_out2 = result['encoder_out2']
    encoder_out3 = result['encoder_out3']
    encoder_out4 = result['encoder_out4']
    inpaint_output = model_inpaint.module.getOutput(encoder_out1, encoder_out2, encoder_out3, encoder_out4,
                                                    add_latent)  # inpaint_output['rec_image4']是8倍插值
    inpaint_output_1x = inpaint_output['rec_image1']  # 未插
    inpaint_output_2x = inpaint_output['rec_image2']
    inpaint_output_4x = inpaint_output['rec_image3']
    inpaint_output_8x = inpaint_output['rec_image4']

    # 修复网络的损失函数
    loss_inpaint_1x = criterion_pixelwise(inpaint_output_1x, image_clean)
    loss_inpaint_2x = criterion_pixelwise(inpaint_output_2x, image_clean)
    loss_inpaint_4x = criterion_pixelwise(inpaint_output_4x, image_clean)
    loss_inpaint_8x = criterion_pixelwise(inpaint_output_8x, image_clean)
    loss_inpaint = 0.5 * loss_inpaint_1x + 0.3 * loss_inpaint_2x + 0.2 * loss_inpaint_4x + 0.1 * loss_inpaint_8x
    # 风格网络的损失函数
    dis_out = dis_model(dot_output)
    # 预测类别
    style_pred = torch.argmax(dis_out, dim=1)
    # 多分类的损失函数
    loss_style = dis_criterion(dis_out, styles_labels_tensor)
    # 计算风格准确率
    # pred = style_pred.to('cpu').detach().tolist()
    # true = styles_labels_tensor.to('cpu').tolist()
    pred = style_pred.detach().tolist()
    true = styles_labels_tensor.tolist()
    dis_acc = accuracy_score(pred, true)
    acc_style.append(dis_acc)
    # 总损失
    total_loss = loss_text + loss_inpaint + loss_style
    if (iteration + 1) % 200 == 0 or (iteration + 1) == iter_len:
        print(
            'epoch : {} | iter : {}/{} | loss_rec : {} | loss_dis : {} | total_loss：{}'.format(epoch + 1, iteration + 1,
                                                                                               iter_len,
                                                                                               loss_rec, loss_dis,
                                                                                               total_loss))
    total_loss.backward()
    model_recog_optimizer.step()
    model_inpaint_optimizer.step()
    dis_optimizer.step()
    # 损失函数的可视化
    writer.add_scalar('total_loss', total_loss.item(), times)
    writer.add_scalar('loss_text', loss_text.item(), times)
    writer.add_scalar('loss_inpaint', loss_inpaint.item(), times)
    writer.add_scalar('loss_style', loss_style.item(), times)
    writer.add_scalar('loss_style_acc', dis_acc, times)

    # 10.9 图像的tensorboard 显示
    show_out_1x = (inpaint_output_1x + 1) / 2
    show_out_clean = (image_clean + 1) / 2
    writer.add_image('original_image_clean', show_out_clean[0], times, dataformats='CHW')
    writer.add_image('inpaint_output_1x', show_out_1x[0], times, dataformats='CHW')

    # writer.flush()
    # 添加损失txt
    if (iteration + 1) % 100 == 0 or (iteration + 1) == iter_len:
        with open(os.path.join('./Log_Loss', 'total_loss.txt'), 'a') as f:
            f.write('epoch : {} | iter : {}/{} | total_loss : {}\n '.format(epoch + 1, iteration + 1, iter_len,
                                                                            total_loss.item()))
        with open(os.path.join('./Log_Loss', 'loss_text.txt'), 'a') as f:
            f.write('epoch : {} | iter : {}/{} | loss_text : {} \n'.format(epoch + 1, iteration + 1, iter_len,
                                                                           loss_text.item()))
        with open(os.path.join('./Log_Loss', 'loss_inpaint.txt'), 'a') as f:
            f.write('epoch : {} | iter : {}/{} | loss_inpaint : {}\n '.format(epoch + 1, iteration + 1, iter_len,
                                                                              loss_inpaint.item()))
        with open(os.path.join('./Log_Loss', 'loss_style.txt'), 'a') as f:
            f.write('epoch : {} | iter : {}/{} | loss_style : {}\n '.format(epoch + 1, iteration + 1, iter_len,
                                                                            loss_style.item()))
        with open(os.path.join('./Log_Loss', 'style_acc.txt'), 'a') as f:
            f.write('epoch : {} | iter : {}/{} | style_acc : {}\n '.format(epoch + 1, iteration + 1, iter_len,
                                                                           dis_acc))
    return acc_style


def get_test_part_label(test_labels):
    test_parts_label = []
    for label in test_labels:
        char = label.split("_")[0]
        test_parts_label.append(char)
    return test_parts_label


test_time = 0


@torch.no_grad()
def test(epoch):
    torch.cuda.empty_cache()
    global test_time
    test_time += 1
    torch.save(model_recog.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

    result_file = open('./history/{}/result_file_test_{}.txt'.format(config['exp_name'], test_time), 'w+',
                       encoding='utf-8')

    print("Start Eval!")
    model_recog.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('test:', test_loader_len)

    correct = 0
    total = 0

    for iteration in range(test_loader_len):
        data = next(dataloader)
        image, label, _ = data
        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
        # 添加 转换label
        test_part_label = get_test_part_label(label)
        length, text_input, text_gt, string_label = converter(test_part_label)
        max_length = max(length)
        batch = image.shape[0]
        pred = torch.zeros(batch, 1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()

        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1
            result = model_recog(image, length_tmp, pred, conv_feature=image_features, test=True)

            prediction = result['pred']
            prediction = prediction / prediction.norm(dim=1, keepdim=True)
            prediction = prediction @ text_features.t()
            pred = torch.max(torch.softmax(prediction, 2), 2)[1]
            prob[:, i] = torch.max(torch.softmax(prediction, 2), 2)[0][:, -1]

            # pred = torch.cat((pred, now_pred.view(-1, 1)), 1)
            # probs, index = prediction.softmax(dim=-1).max(dim=-1)
            # if (probs == prob.cuda()).all():
            #     print("这两种计算方式相同，输出结果相同")
            # image_features = result['conv']
        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                now_pred.append(pred[i][j])
            # text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())
            text_pred_list.append(torch.Tensor(now_pred).long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)

        start = 0
        for i in range(batch):
            state = False
            pred = tensor2str(text_pred_list[i])
            gt = tensor2str(text_gt_list[i])

            if pred == gt:
                correct += 1
                state = True

            start += i
            total += 1
            # print('{} | {} | {} | {} | {} | {}'.format(total, pred, gt, state, text_prob_list[i],
            #                                            correct / total))
            result_file.write(
                '{} | {} | {} | {} | {} \n'.format(total, pred, gt, state, text_prob_list[i]))

    print("ACC : {}".format(correct / total))
    global best_acc

    if correct / total > best_acc:
        best_acc = correct / total
        torch.save(model_recog.state_dict(), './history/{}/best_model.pth'.format(config['exp_name']))

    f = open('./history/{}/record.txt'.format(config['exp_name']), 'a+', encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct / total))
    f.close()
    acc_recog = correct / total
    # 添加返回值
    return acc_recog


if __name__ == '__main__':
    print('-------------')
    if config['test_only']:
        test(-1)
        exit(0)
    # 建立保存网络模型的base路径
    if not os.path.exists(config['save_model_dir']):
        os.makedirs(config['save_model_dir'])
    if not os.path.exists('./history/{}'.format(config['exp_name'])):
        os.makedirs('./history/{}'.format(config['exp_name']))

    # 建立保存损失函数的路径
    if not os.path.exists('./Log_Loss'):
        os.makedirs('./Log_Loss')

    times = 0

    all_epoch_time_start = time.time()
    for epoch in range(config['epoch']):
        torch.save(model_recog.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

        grouped_labels = get_all_label()
        iter_len = len(grouped_labels)
        # dataloader = iter(train_loader)
        # train_loader_len = len(train_loader)
        # print('training:', train_loader_len)
        # for iteration in range(train_loader_len):  # train_loader_len
        one_epoch_time_start = time.time()
        acc_style = list()
        for iteration, (label) in enumerate(zip(grouped_labels)):
            # data = next(dataloader)  # 执行__getitem__
            # image, label, _ = data

            label = list(label[0])
            # 从本地获取mask图像
            image = get_images(label)

            image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))  # 执行插值操作

            # # 从本地（非lmdb）获取干净图像
            image_clean, styles_labels, parts_label = get_images_and_style_index(label)
            # 将styles_labels list形式转成成int
            styles_labels_tensor = torch.tensor(list(map(int, styles_labels))).cuda()
            length, text_input, text_gt, string_label = converter(parts_label)
            acc_style = train(epoch, iteration, image, length, text_input, text_gt, image_clean, styles_labels_tensor,
                              iter_len, acc_style, times)
            times += 1
            #torch.cuda.empty_cache()  # 10.20 添加

        style_dis_acc = np.mean(acc_style)  # 每轮498组的平均
        recog_acc = test(epoch)
        model_recog_scheduler.step()
        model_inpaint_scheduler.step()
        dis_optimizer_scheduler.step()
        if (epoch + 1) % 10 == 0:
            save_path_model = os.path.join(config['save_model_dir'], 'RecogNet_state_' + str(epoch + 1) + '.pt')
            torch.save(model_recog.state_dict(), save_path_model)
            save_path_model = os.path.join(config['save_model_dir'], 'InpaintNet_state_' + str(epoch + 1) + '.pt')
            torch.save(model_inpaint.state_dict(), save_path_model)
            save_path_model = os.path.join(config['save_model_dir'], 'DisNet_state_' + str(epoch + 1) + '.pt')
            torch.save(dis_model.state_dict(), save_path_model)
        one_epoch_time_end = time.time()
        print('This epoch take time {:.5f}'.format(one_epoch_time_end - one_epoch_time_start))
    all_epoch_time_end = time.time()
    print('The all epoch take time {:.5f}'.format(all_epoch_time_end - all_epoch_time_start))
    writer.close()

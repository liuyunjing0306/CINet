# -*- coding: utf-8 -*-
import os
import cv2
import time  # 导入 time 模块
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np


def imread_with_unicode(path):
    # 使用 open() 以二进制模式读取图片数据
    with open(path, 'rb') as f:
        img_data = f.read()
    # 将读取的字节数据转化为 numpy 数组
    img_array = np.frombuffer(img_data, np.uint8)
    # 解码为 OpenCV 图像格式
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def calculate_psnr_ssim(A_path, A1_path, output_file, avg_output_file):
    psnr_list = []
    ssim_list = []
    time_list = []  # 用于存储每张图片的处理时间

    # 打开文件用于写入每张图片的 PSNR 和 SSIM 值
    with open(output_file, 'w') as f:
        f.write("File Name, PSNR, SSIM, Time (s)\n")

        # 遍历 A 文件夹下的所有子文件夹
        for folder in os.listdir(A_path):
            folder_A = os.path.join(A_path, folder)
            folder_A1 = os.path.join(A1_path, folder)

            # 检查给定路径是否是一个有效的目录
            if os.path.isdir(folder_A) and os.path.isdir(folder_A1):
                # 遍历子文件夹中的所有图片文件
                for file_name in os.listdir(folder_A):
                    # 添加
                    file_name=file_name.split(".")[0]
                    file_path_A = os.path.join(folder_A, file_name+".jpg")
                    file_path_A1 = os.path.join(folder_A1, file_name+".png")

                    # 确保两个文件都存在 检查是否是一个有效文件
                    if os.path.isfile(file_path_A) and os.path.isfile(file_path_A1):

                        # 记录开始时间
                        start_time = time.time()

                        # 读取图像（修改位置：使用 imread_with_unicode 替代 cv2.imread）
                        image_A = imread_with_unicode(file_path_A)
                        image_A1 = imread_with_unicode(file_path_A1)

                        # 确保读取成功并且两张图片大小相同
                        if image_A is not None and image_A1 is not None:
                            if image_A.shape == image_A1.shape:
                                # 转换 BGR 到 RGB
                                image_A_rgb = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
                                image_A1_rgb = cv2.cvtColor(image_A1, cv2.COLOR_BGR2RGB)

                                # 计算 PSNR
                                psnr_value = psnr(image_A_rgb, image_A1_rgb, data_range=255)
                                # 计算 SSIM
                                ssim_value, _ = ssim(image_A_rgb, image_A1_rgb, full=True, channel_axis=-1)

                                # 记录 PSNR 和 SSIM
                                psnr_list.append(psnr_value)
                                ssim_list.append(ssim_value)

                                # 记录结束时间并计算所需时间
                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                time_list.append(elapsed_time)

                                # 写入到文件
                                f.write(f"{file_name}, {psnr_value:.5f}, {ssim_value:.5f}, {elapsed_time:.4f}\n")
                            else:
                                print(f"Image size mismatch for {file_name}")
                        else:
                            print(f"Error reading images for {file_name}")
                    else:
                        print(f"File not found: {file_name}")
            else:
                print("A和A1下面的子目录至少有一个目录不存在")

    # 计算平均 PSNR 和 SSIM
    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0
    avg_time = sum(time_list) / len(time_list) if time_list else 0

    # 将平均值写入到另一个文档
    with open(avg_output_file, 'w') as avg_f:
        avg_f.write(f"Average PSNR: {avg_psnr:.5f}\n")
        avg_f.write(f"Average SSIM: {avg_ssim:.5f}\n")
        avg_f.write(f"Average Time: {avg_time:.5f} seconds\n")  # 写入平均时间


# 调用函数
A_path ='D://LYJ_Project//Project_Differ_Mask//Mask_0.4//ProductImage//original_data//test1'  # 原始图像
A1_path = 'D://LYJ_Project//Project_Differ_Mask//Mask_0.4//My_Test//Use_Lmdb_Date_Test//Test_Results//InpaintNet'  # 恢复后的图像
output_file = 'psnr_ssim_values.txt'  # 保存每张图片PSNR和SSIM的文件
avg_output_file = 'average_psnr_ssim.txt'  # 保存平均PSNR和SSIM的文件

calculate_psnr_ssim(A_path, A1_path, output_file, avg_output_file)

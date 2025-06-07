import os
import torch
from pytorch_fid import fid_score


# 计算FID的函数，使用Inception V3
def calculate_fid_inception(folder_A, folder_A1, output_file):
    print("Calculating FID using Inception...")

    # 使用pytorch-fid库计算FID
    fid_value = fid_score.calculate_fid_given_paths([folder_A, folder_A1],
                                                    batch_size=50,
                                                    device='cuda',  # 如果有GPU可以用'cuda'
                                                    dims=2048)  # InceptionV3池化层输出维度为2048

    print(f"FID (Inception): {fid_value}")

    # 保存FID结果到txt文件
    with open(output_file, 'w') as file:
        file.write(f"FID (Inception): {fid_value}\n")

    print(f"FID result saved to {output_file}")


if __name__ == '__main__':
    # 设置文件夹路径
    folder_A = './original_image'  # 原始图像
    folder_A1 = './restored_image'  # 修复图像

    # 设置保存结果的文件路径
    output_file = 'inception_fid_results.txt'

    # 计算FID并保存结果
    calculate_fid_inception(folder_A, folder_A1, output_file)

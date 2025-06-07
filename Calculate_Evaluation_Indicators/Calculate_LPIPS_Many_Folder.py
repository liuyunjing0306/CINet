import os
import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms
import time

# 加载LPIPS损失函数
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_squeeze = lpips.LPIPS(net='squeeze')


# 加载图像并转换为张量
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    return transform(img).unsqueeze(0)


# 计算并输出LPIPS距离
def compute_lpips(image1, image2, loss_fn):
    distance = loss_fn(image1, image2)
    return distance.item()


# 遍历两个文件夹，计算LPIPS距离，并记录时间
def calculate_lpips_for_folders(folder_A, folder_A1, per_image_file, summary_file):
    results = []
    total_time = 0
    num_images = 0

    # 写入每张图片的 LPIPS 结果到 txt 文件
    with open(per_image_file, 'w') as file:
        file.write("filename\tlpips_alex\tlpips_vgg\tlpips_squeeze\ttime_taken\n")

        # 遍历A文件夹中的所有子文件夹
        for subfolder in os.listdir(folder_A):  # 修改1：遍历A文件夹中的子文件夹
            subfolder_path_A = os.path.join(folder_A, subfolder)
            subfolder_path_A1 = os.path.join(folder_A1, subfolder)  # 修改2：对应的A1文件夹的子文件夹

            # 确保子文件夹路径是有效的
            if os.path.isdir(subfolder_path_A) and os.path.isdir(subfolder_path_A1):  # 修改3：检查是否为文件夹
                print(f"Processing subfolder: {subfolder}")  # 打印当前处理的子文件夹

                for filename in os.listdir(subfolder_path_A):  # 修改4：遍历子文件夹中的图像
                    # 添加
                    filename = filename.split(".")[0]
                    img_A_path = os.path.join(subfolder_path_A, filename + ".jpg")
                    img_A1_path = os.path.join(subfolder_path_A1, filename + ".png")

                    # img_A_path = os.path.join(subfolder_path_A, filename)
                    # img_A1_path = os.path.join(subfolder_path_A1, filename)

                    # 确保在两个文件夹中都有同名文件
                    if os.path.exists(img_A_path) and os.path.exists(img_A1_path):
                        print(f"Calculating LPIPS for: {filename}")  # 打印当前计算的图像
                        img_A = load_image(img_A_path)
                        img_A1 = load_image(img_A1_path)

                        start_time = time.time()  # 开始计时

                        # 计算三种模型的LPIPS
                        lpips_alex = compute_lpips(img_A, img_A1, loss_fn_alex)
                        lpips_vgg = compute_lpips(img_A, img_A1, loss_fn_vgg)
                        lpips_squeeze = compute_lpips(img_A, img_A1, loss_fn_squeeze)

                        time_taken = time.time() - start_time  # 计算时间
                        total_time += time_taken
                        num_images += 1

                        # 记录每张图片的 LPIPS 值和时间到文件中
                        file.write(
                            f"{filename}\t{lpips_alex:.6f}\t{lpips_vgg:.6f}\t{lpips_squeeze:.6f}\t{time_taken:.6f}\n")

                        # 将每张图像的结果保存到列表中
                        results.append({
                            'lpips_alex': lpips_alex,
                            'lpips_vgg': lpips_vgg,
                            'lpips_squeeze': lpips_squeeze,
                            'time_taken': time_taken
                        })

    # 计算平均 LPIPS 和平均时间
    if num_images > 0:  # 防止除以零
        avg_lpips_alex = sum(r['lpips_alex'] for r in results) / num_images
        avg_lpips_vgg = sum(r['lpips_vgg'] for r in results) / num_images
        avg_lpips_squeeze = sum(r['lpips_squeeze'] for r in results) / num_images
        avg_time_taken = total_time / num_images

        # 写入平均结果到另一个 txt 文件
        with open(summary_file, 'w') as file:
            file.write("Average LPIPS and Time Results\n")
            file.write(f"Average LPIPS (AlexNet): {avg_lpips_alex:.6f}\n")
            file.write(f"Average LPIPS (VGG): {avg_lpips_vgg:.6f}\n")
            file.write(f"Average LPIPS (SqueezeNet): {avg_lpips_squeeze:.6f}\n")
            file.write(f"Average time per image: {avg_time_taken:.6f} seconds\n")

        # 输出计算总结果
        print(f'Average LPIPS (AlexNet): {avg_lpips_alex}')
        print(f'Average LPIPS (VGG): {avg_lpips_vgg}')
        print(f'Average LPIPS (SqueezeNet): {avg_lpips_squeeze}')
        print(f'Average time per image: {avg_time_taken} seconds')
    else:
        print("No images were processed.")


# 设置文件夹路径
folder_A = 'D://LYJ_Project//Project_Differ_Mask//Mask_0.4//ProductImage//original_data//test1'  # 原始图像
folder_A1 = 'D://LYJ_Project//Project_Differ_Mask//Mask_0.4//My_Test//Use_Lmdb_Date_Test//Test_Results//InpaintNet'  # 修复图像

# 设置保存结果的文件路径
per_image_file = 'lpips_per_image.txt'  # 保存每张图像LPIPS和时间
summary_file = 'lpips_summary.txt'  # 保存平均LPIPS和平均时间

# 计算LPIPS并记录结果
calculate_lpips_for_folders(folder_A, folder_A1, per_image_file, summary_file)

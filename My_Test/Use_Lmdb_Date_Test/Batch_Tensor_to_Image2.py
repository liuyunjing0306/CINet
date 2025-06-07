### torchvision.utils 保存
import os
from config import config
from torchvision.utils import save_image
save_dir_restored_image = "./Test_Results"


# 假设 resizeNormalize 和网络推理函数已经定义
# 添加 转换label
def get_test_split_label(label):
    test_parts_label = []
    test_name_label = []
    test_styles_indexs = []  # 风格的标签
    #for label in test_labels:
    char = label.split("_")[0]
    name = label.split("_")[1]
    style_index = label.split("_")[2]
    test_parts_label.append(char)
    test_name_label.append(name)
    test_styles_indexs.append(style_index)

    return test_parts_label, test_name_label, test_styles_indexs


# 恢复图像的函数
def restore_image(output_tensor, label):
    # 取消标准化: 逆转 (img - 0.5) / 0.5 -> 回到 [0, 1]
    # output_tensor = output_tensor.mul(0.5).add(0.5).clamp(0, 1)
    for i in range(output_tensor.size(0)):
        output = output_tensor[i]
        output = (output + 1) / 2  # 修改
        output = output.clamp(0, 1)
        # # 转换为 NumPy 数组并调整到 [0, 255] 范围
        # # output_array = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
        # output_array = output.permute(1, 2, 0).cpu().numpy()
        # output_array = (output_array * 255).astype(np.uint8)
        # # 将 NumPy 数组转换为 PIL 图像
        # output_image = Image.fromarray(output_array)
        test_content_label, test_name_label, test_styles_indexs = get_test_split_label(label[i])
        test_content_label_str = ', '.join(map(str, test_content_label))
        test_name_label_str = ', '.join(map(str, test_name_label))
        test_styles_indexs_str = ', '.join(map(str, test_styles_indexs))
        # 调整图像大小为原始尺寸
        # output_image = output_image.resize(original_size, Image.BILINEAR)
        #保存图像
        save_single_image(output, save_dir_restored_image, test_content_label_str, test_name_label_str,
                   test_styles_indexs_str)


# 保存单张图像的函数
def save_single_image(output_image_tensor, save_dir, test_parts_label, test_name_label, test_styles_indexs):
    # output_image = output_image.convert("RGB")
    save_path = os.path.join(save_dir, config['exp_name_inpaint'], test_parts_label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f"{test_name_label}_{test_styles_indexs}.png"  # 将jpg修改成png

    full_save_path = os.path.join(save_path, file_name)
    save_image(output_image_tensor, full_save_path)
    # 打印保存路径和图片
    print(f"Image saved to: {full_save_path}")

import os
import argparse
from collections import OrderedDict
import time  # 导入 time 

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.CSDehazeNet import CSDehaze  

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='CSDehaze', type=str, help='模型名称')
parser.add_argument('--data_dir', default='./hazy', type=str, help='待推理的图像目录')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='保存模型权重的目录')
parser.add_argument('--result_dir', default='./out/', type=str, help='保存推理结果的目录')
args = parser.parse_args()


def single(save_dir):
    """加载模型"""
    state_dict = torch.load(save_dir, map_location=torch.device('cpu'))['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  
        new_state_dict[name] = v
    return new_state_dict


def preprocess_image(image_path):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  


def postprocess_output(output_tensor):
    """输出后处理"""
    img = output_tensor.squeeze(0).detach()  
    img = img * 0.5 + 0.5  # 反归一化
    return transforms.ToPILImage()(img)


def infer_batch(data_dir, model_path, result_dir):
    """批量执行推理并将结果保存到指定路径"""
    print(f"Start testing:")

    # 加载模型
    network = CSDehaze()
    network.load_state_dict(single(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()

    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 获取所有支持格式的图像文件
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(image_files)
    print(f"共找到 {total_images} 张图像，开始处理...")

    # 初始化总处理时间和预热迭代次数
    total_processing_time = 0
    warmup_iterations = min(5, total_images)  # 前5次迭代作为预热，不计算FPS

    # 遍历目录中的所有图像文件
    for idx, filename in enumerate(image_files, start=1):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(data_dir, filename)

            # 构造输出文件名
            name, ext = os.path.splitext(filename)  # 分离文件名和扩展名
            output_filename = f"{name}{ext}"  # 添加 _output 后缀
            output_path = os.path.join(result_dir, output_filename)

            # 预处理输入图像
            input_tensor = preprocess_image(image_path).to(device)

            # 确保GPU预热（如果使用GPU）
            if idx == 1 and device.type == 'cuda':
                with torch.no_grad():
                    _ = network(input_tensor)
                torch.cuda.synchronize()  # 等待所有GPU操作完成

            # 记录开始时间
            start_time = time.time()

            with torch.no_grad():
                output_tensor = network(input_tensor)

            if device.type == 'cuda':
                torch.cuda.synchronize()  # 确保GPU操作完成后再计时

            # 记录结束时间并计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time

            # 只在预热迭代后计算总时间
            if idx > warmup_iterations:
                total_processing_time += processing_time

            # 后处理输出
            output_image = postprocess_output(output_tensor.cpu())
            output_image.save(output_path)

            # 打印处理时间和文件名
            print(f"正在处理第 {idx}/{total_images} 张图像: {filename} Time: {processing_time:.3f} 秒")

    # 计算FPS
    if total_images > warmup_iterations:
        effective_images = total_images - warmup_iterations
        fps = effective_images / total_processing_time
        times = 1/fps
        print(f"\n模型 FPS 计算结果:")
        print(f"总有效处理时间: {total_processing_time:.3f} 秒")
        print(f"有效处理图像数量: {effective_images} 张")
        print(f"平均 FPS: {fps:.2f} 帧/秒")
        print(f"平均 time: {times:.4f} 秒/帧")
    else:
        print("图像数量不足，无法准确计算FPS")

    print(f"所有图像处理完成，结果已保存至 {result_dir}")


if __name__ == '__main__':
    # saved_model_dir = os.path.join(args.save_dir, args.model + '.pth')
    saved_model_dir = os.path.join(args.save_dir, 'epoch_140_psnr_29.36_model.pth')

    if os.path.exists(saved_model_dir):
        print(f"当前模型 {args.model}")
        infer_batch(data_dir=args.data_dir, model_path=saved_model_dir, result_dir=args.result_dir)
    else:
        print("未找到模型，请检查路径是否正确！")

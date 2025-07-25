import os
import argparse
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from models.CSDehazeNet import CSDehaze
from utils import write_img, chw_to_hwc
from PIL import Image
import numpy as np
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='CSDehaze', type=str, help='model name')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--input_dir', default='./hazy', type=str, help='path to input images')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./out', type=str, help='path to results saving')
args = parser.parse_args()


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def test(input_dir, network, result_dir):
    torch.cuda.empty_cache()
    network.eval()

    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])

    # 获取输入目录中所有图像文件
    image_files = [f for f in os.listdir(input_dir) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"在 {input_dir} 目录中未找到图像文件")
        return

    for idx, filename in enumerate(image_files):
        # 读取图像
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')

        # 预处理
        input_tensor = transform(img).unsqueeze(0).cuda()  # 添加批次维度并移至GPU

        with torch.no_grad():
            output = network(input_tensor).clamp_(-1, 1)
            output = output * 0.5 + 0.5  # 转换回[0, 1]范围

        # 保存结果
        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, filename), out_img)

        print(f"Processed {idx + 1}/{len(image_files)}: {filename} saved to {result_dir}")

    print(f"\nAll images saved to {result_dir}")


if __name__ == '__main__':
    # 初始化网络
    network = eval(args.model.replace('-', '_'))()
    network.cuda()

    # 加载模型
    saved_model_dir = os.path.join(args.save_dir, 'CSDehaze_model.pth')
    if not os.path.exists(saved_model_dir):
        print("==> No existing trained model!")
        exit(0)

    network.load_state_dict(single(saved_model_dir))
    print(f"==> Loaded model from {saved_model_dir}")

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在: {args.input_dir}")
        exit(0)

    # 运行测试
    test(args.input_dir, network, args.result_dir)

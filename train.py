import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
import os
from email.mime.text import MIMEText
from email.header import Header
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile
from models.CSDehazeNet import CSDehaze

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='CSDehaze', type=str, help='model name')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data2/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./4k_logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='4K', type=str, help='dataset name')
parser.add_argument('--exp', default='4K', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr  # 导入 formataddr


def send_email(subject, content, sender_email, sender_password, receiver_email):
    try:
        # 构造邮件
        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = formataddr(["训练通知", sender_email])
        message['To'] = formataddr(["接收者", receiver_email])
        message['Subject'] = subject
        # 使用 QQ 邮箱 SMTP 服务发送邮件
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [receiver_email], message.as_string())
        server.quit()
        print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败: {e}")


from pytorch_msssim import ssim


def train(train_loader, network, criterion, optimizer, scaler, epoch, setting, ssim_weight=0.5):
    from torch.cuda.amp import autocast  # 确保autocast可用
    from tqdm import tqdm
    import torch
    from torch.nn.functional import interpolate as F_interpolate

    losses = AverageMeter()
    ssim_losses = AverageMeter()  # 新增SSIM损失记录
    total_losses = AverageMeter()  # 新增总损失记录
    torch.cuda.empty_cache()
    network.train()
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{setting['epochs']} Training", leave=True)

    for batch in train_iter:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(not args.no_autocast):
            output = network(source_img)

            # 计算主损失
            main_loss = criterion(output, target_img)  # 只计算前3个通道的损失
            # 计算SSIM损失，并应用权重
            ssim_loss = (1 - ssim(output, target_img, data_range=1, size_average=True)) * ssim_weight
            total_loss = main_loss + ssim_loss  # 添加加权后的SSIM损失

        losses.update(main_loss.item())
        ssim_losses.update(ssim_loss.item())  # 更新SSIM损失
        total_losses.update(total_loss.item())  # 更新总损失
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_iter.set_postfix(total_loss=total_losses.avg)  # 显示所有损失
    # time.sleep(0.2)  #
    return losses.avg  # 返回总损失、原始损失和SSIM损失平均值


def valid(val_loader, network, best_psnr, sender_email, sender_password, receiver_email, epoch, send_email_flag=True):
    PSNR = AverageMeter()
    network.eval()
    val_iter = tqdm(val_loader, desc=f"          Validating", leave=True)
    for batch in val_iter:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        with torch.no_grad():
            output = network(source_img)
            output = output.clamp_(-1, 1)
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean().item()  # 计算PSNR
        PSNR.update(psnr, source_img.size(0))  # 更新PSNR的平均值
        val_iter.set_postfix(psnr=PSNR.avg)

    # 如果有新的最佳 PSNR，并且开启了发送邮件开关，发送邮件通知
    if PSNR.avg > best_psnr and send_email_flag:
        subject = f"来自台式：第 {epoch} 轮训练：新的最佳 PSNR: {PSNR.avg:.3f}"
        content = (
            f"在第 {epoch} 轮训练后，验证集中获得新的最佳 PSNR 值为 {PSNR.avg:.3f}，"
            f"超过了之前的 {best_psnr:.3f}。"
        )
        try:
            send_email(subject, content, sender_email, sender_password, receiver_email)
            print(f"新的最佳 PSNR: {PSNR.avg:.3f} 邮件通知已发送。")
        except Exception as e:
            print(f"发送新最佳 PSNR 通知邮件失败: {e}")

    return PSNR.avg


# 替换原来的count_macs函数
def count_macs(model, input_shape=(1, 3, 256, 256)):
    input = torch.randn(input_shape).cuda()
    macs, _ = profile(model.module, inputs=(input,), verbose=False)

    # 清除thop添加的临时属性
    for m in model.modules():
        if hasattr(m, "total_ops"):
            del m.total_ops
        if hasattr(m, "total_params"):
            del m.total_params

    return macs


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # send_email_flag = True  # 设置为 True 则发送邮件，False 则不发送邮件
    send_email_flag = False  # 设置为 True 则发送邮件，False 则不发送邮件

    best_psnrs = []  # 用于存储所有的最佳PSNR值
    best_model_paths = []  # 用于存储所有最佳模型的路径

    start_time = time.time()  # 记录最开始的时间

    # 初始化网络、优化器等
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()
    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])

    scheduler = CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()
    dataset_dir = os.path.join(args.data_dir)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    latest_model_path = os.path.join(save_dir, 'latest_model.pth')  # 新增：最新模型路径

    # print(network)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(network):,} trainable parameters.')

    macs1 = count_macs(network, input_shape=(1, 3, (512), (920)))
    macs2 = count_macs(network, input_shape=(1, 3, (256), (256)))
    print(f"MACs: {macs1 / 1e9:.2f} G")
    print(f"MACs: {macs2 / 1e9:.2f} G")

    if os.path.exists(checkpoint_path):
        print('==> Resuming from checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']

        # 如果存在已保存的开始时间，则使用该时间戳
        if 'start_time' in checkpoint:
            start_time = checkpoint['start_time']

    else:
        print('==> Start training, current model name: ' + args.model)
        start_epoch = 1
        best_psnr = 0

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir))

    # 邮件通知
    sender_email = "***************"
    sender_password = "****************"
    receiver_email = "*****************"

    for epoch in range(start_epoch, setting['epochs'] + 1):
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        train_loss = train(train_loader, network, criterion, optimizer, scaler, epoch, setting, ssim_weight=0.2)
        writer.add_scalar('train_loss', train_loss, epoch)
        scheduler.step()

        psnr_for_save = 0  # 初始化用于保存的PSNR值
        if epoch % setting['eval_freq'] == 0:
            avg_psnr = valid(val_loader, network, best_psnr, sender_email, sender_password, receiver_email, epoch,
                             send_email_flag)
            psnr_for_save = avg_psnr  # 如果该轮进行了验证，使用验证得到的PSNR值
            writer.add_scalar('valid_psnr', avg_psnr, epoch)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_psnrs.append(best_psnr)  # 记录新的最佳PSNR值
                new_best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch}_{best_psnr:.2f}.pth")
                torch.save({'state_dict': network.state_dict()}, new_best_model_path)
                best_model_paths.append(new_best_model_path)  # 记录新的最佳模型路径
                print(f"New best PSNR: {best_psnr:.2f}.")
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)

                best_model_path = new_best_model_path

            writer.add_scalar('best_psnr', best_psnr, epoch)

        # 每一轮都保存权重，文件名包含轮次信息和PSNR值
        round_model_path = os.path.join(save_dir, f"epoch_{epoch}_psnr_{psnr_for_save:.2f}_model.pth")
        torch.save({'state_dict': network.state_dict()}, round_model_path)

        # 保存检查点（包含更多训练状态信息）
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_psnr': best_psnr,
            'start_time': start_time,
            'train_loss': train_loss
        }, checkpoint_path)

        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        remaining_epochs = setting['epochs'] - epoch
        remaining_time = remaining_epochs * epoch_duration  # 将时间转换为小时和分钟

        def format_time(seconds):
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{hours:.0f}h {minutes:.0f}min {seconds:.2f}s"


        epoch_duration_str = format_time(epoch_duration)
        elapsed_time_str = format_time(elapsed_time)
        remaining_time_str = format_time(remaining_time)
        estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)
        estimated_completion_str = estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')
        # 打印信息
        print(f"Epoch {epoch}/{setting['epochs']} completed| "
              f"剩余时间: {remaining_time_str} |"
              f"预计完成时间: {estimated_completion_str}.")
        # time.sleep(0.2)

    writer.close()

    end_time = time.time()  # 训练结束后记录结束时间
    total_duration = end_time - start_time  # 计算总耗时

    def format_time(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:.0f}h {minutes:.0f}min {seconds:.2f}s"


    total_duration_str = format_time(total_duration)
    print(f"Total training duration: {total_duration_str}")

    print(f" Best PSNRs: {best_psnrs}")
    print(f" Best model paths: {best_model_paths}")

    subject = f"来自台式：训练结束通知：模型 {args.model}"
    content = (f"训练已结束！\n"
               f"最佳PSNRs：{best_psnrs}\n"
               f"最佳模型路径：{best_model_paths}\n"
               f"总训练时间：{total_duration_str}\n")
    if send_email_flag:
        try:
            send_email(subject, content, sender_email, sender_password, receiver_email)
            print("训练结束通知邮件已发送！")
        except Exception as e:
            print(f"发送训练结束通知邮件失败: {e}")
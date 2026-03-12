import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 你的核心想法：一致性投影优化器
# ==========================================
class ConsistencySGD(optim.Optimizer):
    """
    实现了“迭代式一致性投影”算法。
    
    逻辑：
    维护一个历史动量方向 M (buffer)。
    对于当前梯度 g，计算它在 M 上的投影： g_proj = (g · M / ||M||^2) * M
    最终更新方向是 g_proj 和原始 g 的加权混合。
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, projection_beta=0.8, 
                 weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, projection_beta=projection_beta,
                        weight_decay=weight_decay)
        super(ConsistencySGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            proj_beta = group['projection_beta']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad # 当前梯度 g_t
                
                # 权重衰减
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                state = self.state[p]
                
                # 初始化动量 Buffer (M_t)
                if 'momentum_buffer' not in state:
                    buf = d_p.clone().detach()
                    state['momentum_buffer'] = buf
                    # 第一步没有历史，直接使用当前梯度
                    update_direction = d_p
                else:
                    buf = state['momentum_buffer']
                    
                    # --- 核心逻辑 Start ---
                    # 计算当前梯度 d_p 在历史方向 buf 上的投影
                    
                    # 1. 拉平计算点积 (Layer-wise Projection)
                    g_flat = d_p.view(-1)
                    b_flat = buf.view(-1)
                    
                    dot_prod = torch.dot(g_flat, b_flat)
                    buf_norm_sq = torch.dot(b_flat, b_flat) + 1e-10 # 防止除零
                    
                    # 2. 计算投影系数 scalar = (g . m) / ||m||^2
                    proj_scalar = dot_prod / buf_norm_sq
                    
                    # 3. 得到投影向量
                    grad_projected = proj_scalar * buf
                    
                    # 4. 混合策略
                    # 如果 g 和 buf 方向一致，grad_projected 会增强
                    # 如果 g 和 buf 垂直或反向，grad_projected 会归零或反向修正
                    update_direction = proj_beta * grad_projected + (1 - proj_beta) * d_p
                    # --- 核心逻辑 End ---
                    
                    # 更新历史动量 Buffer (标准 Momentum 更新)
                    buf.mul_(momentum).add_(d_p, alpha=1.0)

                # 应用更新
                p.add_(update_direction, alpha=-lr)

        return loss

# ==========================================
# 2. 测试环境 (Simple MNIST Model)
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(optimizer_name, device, train_loader, steps=200):
    model = Net().to(device)
    lr = 0.01
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        # 使用你的优化器，beta=0.8
        optimizer = ConsistencySGD(model.parameters(), lr=lr, momentum=0.9, projection_beta=0.8)
        
    model.train()
    losses = []
    
    print(f"Training with {optimizer_name}...")
    
    step_count = 0
    # 简单的循环，只要达到指定步数就停止
    while step_count < steps:
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            step_count += 1
            if step_count >= steps:
                break
                
    return losses

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 下载数据
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 运行实验 (跑 300 个 Batch 看收敛速度)
    TOTAL_STEPS = 300
    loss_sgd = train('SGD', device, train_loader, steps=TOTAL_STEPS)
    loss_custom = train('Consistency', device, train_loader, steps=TOTAL_STEPS)

    # 平滑曲线以便观察
    def smooth(scalars, weight=0.8):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    smooth_sgd = smooth(loss_sgd)
    smooth_custom = smooth(loss_custom)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(loss_sgd, color='blue', alpha=0.2)
    plt.plot(loss_custom, color='red', alpha=0.2)
    plt.plot(smooth_sgd, label='Standard SGD (Momentum=0.9)', color='blue', linewidth=2)
    plt.plot(smooth_custom, label='Consistency Projection (Yours)', color='red', linewidth=2)
    
    plt.title(f'Optimizer Comparison (First {TOTAL_STEPS} Steps)')
    plt.xlabel('Batch Steps')
    plt.ylabel('NLL Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Done! Check the plot.")
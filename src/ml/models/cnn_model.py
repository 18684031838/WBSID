"""
CNN模型实现
"""
import numpy as np
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .base_model import BaseModel

class CNNNet(nn.Module):
    def __init__(self, input_dim: int):
        super(CNNNet, self).__init__()
        # 简单而有效的网络架构
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            # 第二个卷积块
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            # 第三个卷积块，减少特征维度
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25)
        )
        
        # 计算全连接层的输入维度
        self.fc_input_dim = 32 * ((input_dim + 3) // 8)  # 经过3次MaxPool1d，特征维度变为原来的1/8
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # 添加通道维度 [batch_size, 1, input_dim]
        x = x.unsqueeze(1)
        
        # 特征提取
        x = self.features(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return torch.softmax(x, dim=1)

class CNNModel(BaseModel):
    def __init__(self, input_dim: int = 19):
        """初始化CNN模型
        
        Args:
            input_dim: 输入特征维度，默认为19（TF-IDF特征9维 + 统计特征10维）
        """
        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 创建模型和优化器
        self.model = CNNNet(input_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 训练参数
        self.batch_size = 64
        self.epochs = 10
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """适配scikit-learn的fit接口，映射到train方法"""
        return self.train(X, y)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # 数据预处理：标准化
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / (std + 1e-8)
        
        # 转换为PyTorch张量并移动到GPU
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0  # Windows下多进程可能有问题，设为0
        )
        
        # 训练模型
        best_accuracy = 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # 使用tqdm显示进度条
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for batch_X, batch_y in pbar:
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # 更新进度条
                total_loss += loss.item()
                avg_loss = total_loss / (pbar.n + 1)
                accuracy = 100 * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'accuracy': f'{accuracy:.2f}%'
                })
            
            # 保存最佳模型
            if correct / total > best_accuracy:
                best_accuracy = correct / total
                best_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': best_accuracy
                }
            
            pbar.close()
        
        # 恢复最佳模型
        self.model.load_state_dict(best_state['model_state_dict'])
        self.optimizer.load_state_dict(best_state['optimizer_state_dict'])
        print(f'\nBest accuracy: {best_accuracy*100:.2f}%')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            # 数据预处理：标准化
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / (std + 1e-8)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            # 数据预处理：标准化
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / (std + 1e-8)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'model_structure': str(self.model),
            'device': str(self.device)
        }
    
    def save(self, path: str) -> None:
        """保存模型状态"""
        # 如果路径已经有扩展名，先移除
        if '.' in path.split('/')[-1]:
            path = path.rsplit('.', 1)[0]
        
        # 添加正确的扩展名
        path = path + self.model_extension
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }, path)
    
    def load(self, path: str) -> None:
        """加载模型状态"""
        # 如果路径已经有扩展名，先移除
        if '.' in path.split('/')[-1]:
            path = path.rsplit('.', 1)[0]
        
        # 添加正确的扩展名
        path = path + self.model_extension
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.epochs = checkpoint['epochs']
        self.model.eval()
        
    @property
    def model_extension(self) -> str:
        """返回模型文件扩展名"""
        return '.pt'

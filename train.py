import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# 데이터셋 정의
class SegmentationDataset(Dataset):
    def __init__(self, rgb_dir, seg_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.seg_dir = seg_dir
        self.transform = transform

        self.rgb_images = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        self.seg_images = sorted(glob.glob(os.path.join(seg_dir, '*.npy')))

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_images[idx]).convert('RGB')
        seg_image = np.load(self.seg_images[idx])
                
        # 마지막 차원을 없애 2D 형식으로 변환
        seg_image = np.argmax(seg_image, axis=-1)
        
        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, seg_image

# 모델 정의
class SimpleFCN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleFCN, self).__init__()
        
        # Encoder (Downsampling)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder (Upsampling)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Convolution layers after concatenation (skip connections)
        self.conv3_dec = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2_dec = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv1_dec = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder path
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(self.maxpool(x1))))
        x3 = self.relu(self.bn3(self.conv3(self.maxpool(x2))))
        x4 = self.relu(self.bn4(self.conv4(self.maxpool(x3))))
        
        # Decoder path with skip connections
        d3 = self.relu(self.deconv3(x4))
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.relu(self.conv3_dec(d3))
        
        d2 = self.relu(self.deconv2(d3))
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.relu(self.conv2_dec(d2))
        
        d1 = self.relu(self.deconv1(d2))
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.relu(self.conv1_dec(d1))
        
        # Final output layer
        output = self.final_conv(d1)
        
        return output        
                
# 하이퍼파라미터 및 설정
batch_size = 16
num_classes = 3
learning_rate = 0.001
num_epochs = 50
rgb_img_dir = 'rgb_images'
seg_img_dir = 'segmentation_images'
val_ratio = 0.2

# Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 및 DataLoader 정의
dataset = SegmentationDataset(rgb_dir=rgb_img_dir, seg_dir=seg_img_dir, transform=transform)

dataset_size = len(dataset)
val_size = int(val_ratio * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 옵티마이저 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleFCN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 검증 함수
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device).long().squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.numel()
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return val_loss / len(val_loader), accuracy

best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        images = images.to(device)
        targets = targets.to(device).long().squeeze(1)
    
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    val_loss, val_accuracy = validate(model, val_loader, criterion)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    # 최선의 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_fcn_segmentation.pth')
        print("Best model saved!")

print("Training completed!")
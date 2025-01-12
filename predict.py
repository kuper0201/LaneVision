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

# 모델 로드 함수
def load_model(model_path):
    model = SimpleFCN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def save_prediction(model, image_path, save_path):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        output = model(image_tensor)
        output = F.softmax(output, dim=1)
        output = output.squeeze(0)
        
        prediction = output.argmax(dim=0).cpu().numpy()
        
        red_channel = np.zeros_like(prediction, dtype=np.uint8)
        blue_channel = np.zeros_like(prediction, dtype=np.uint8)
        green_channel = np.zeros_like(prediction, dtype=np.uint8)
        
        green_channel[prediction == 0] = 0
        red_channel[prediction == 1] = 255
        blue_channel[prediction == 2] = 255

        # RGB 이미지 생성
        rgb_image = np.stack([blue_channel, green_channel, red_channel], axis=-1)
        cv2.imwrite(save_path, rgb_image)

def predict_directory(model, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 출력 디렉터리가 없으면 생성

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    image_files = image_files[:1000]
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_dir, image_file)
        save_image_path = os.path.join(output_dir, image_file)  # 동일한 파일명으로 저장
        
        # 예측 및 결과 저장
        save_prediction(model, image_path, save_image_path)

# Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 예측 실행
model = load_model('best_fcn_segmentation.pth')

# 입력 디렉터리와 출력 디렉터리 지정
input_directory = 'rgb_images'  # 원본 이미지가 있는 디렉터리
output_directory = 'predicted_images'  # 예측 결과를 저장할 디렉터리

predict_directory(model, input_directory, output_directory)
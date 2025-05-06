import os
import torch
import cv2
import numpy as np
import face_recognition
from torchvision import models, transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Define the Model Class
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Dataset Class
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=20, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = 1 if self.labels[idx] == "FAKE" else 0
        frames = []
        for frame in self.extract_frames(video_path):
            faces = face_recognition.face_locations(frame)
            if faces:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            frames.append(self.transform(frame))
            if len(frames) == self.sequence_length:
                break
        return torch.stack(frames)[:self.sequence_length].unsqueeze(0), label

    def extract_frames(self, video_path):
        vidObj = cv2.VideoCapture(video_path)
        success, image = vidObj.read()
        while success:
            yield image
            success, image = vidObj.read()

# Load Test Videos
real_videos = [os.path.join("D:/deepfake-detection/Backend/TEST/real", f) for f in os.listdir("D:/deepfake-detection/Backend/TEST/real") if f.endswith(".mp4")]
fake_videos = [os.path.join("D:/deepfake-detection/Backend/TEST/fake", f) for f in os.listdir("D:/deepfake-detection/Backend/TEST/fake") if f.endswith(".mp4")]

test_videos = real_videos + fake_videos
test_labels = ["REAL"] * len(real_videos) + ["FAKE"] * len(fake_videos)

# Define Transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create Dataset and DataLoader
dataset = VideoDataset(test_videos, test_labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load Model
model = Model(2)
model.load_state_dict(torch.load("G:/deepfake-detection/model/df_model.pt", map_location=torch.device('cpu')))
model.eval()

# Evaluate Model
def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    sm = nn.Softmax(dim=1)
    
    for i, (inputs, labels) in enumerate(dataloader):
        fmap, logits = model(inputs)
        logits = sm(logits)
        predicted = torch.argmax(logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = (correct / total) * 100
    print(f"Model Accuracy: {accuracy:.2f}%")

# Run Evaluation
evaluate_model(model, dataloader)

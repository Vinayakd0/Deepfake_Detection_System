from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
import cv2
import numpy as np
import face_recognition
from torchvision import models, transforms
from torch import nn
from torch.utils.data import Dataset
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={r"/Detect": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'Uploaded_Files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Real-ESRGAN Model
def load_esrgan_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=r"D:\deepfake-detection\Backend\weights\RealESRGAN_x4.pth",  # Make sure this exists
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )
    return upsampler

esrgan = load_esrgan_model()

# Model Definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
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
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=20, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for frame in self.frame_extract(video_path):
            try:
                faces = face_recognition.face_locations(frame)
                top, right, bottom, left = faces[0]
                face_crop = frame[top:bottom, left:right, :]

                # Convert to BGR for ESRGAN
                face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                sr_face, _ = esrgan.enhance(face_bgr, outscale=4)
                sr_face = cv2.cvtColor(sr_face, cv2.COLOR_BGR2RGB)

                transformed = self.transform(sr_face)
            except Exception as e:
                # fallback if no face or error
                transformed = self.transform(frame)

            frames.append(transformed)
            if len(frames) == self.count:
                break
        return torch.stack(frames)[:self.count].unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# Prediction Function
def predict(model, img):
    sm = nn.Softmax(dim=1)
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

# Deepfake Detection Function
def detect_fake_video(video_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ValidationDataset([video_path], transform=transform)
    model = Model(2)
    model_path = "G:/deepfake-detection/model/df_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    prediction, confidence = predict(model, dataset[0])
    return prediction, confidence

# Routes
@app.route('/')
def home():
    return "Deepfake Detection API is Running"

@app.route('/Detect', methods=['POST'])
def Detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    try:
        print(f"Processing video: {video_path}")
        prediction, confidence = detect_fake_video(video_path)
        print(f"Prediction result: {prediction}")

        result = "FAKE" if prediction == 0 else "REAL"
        os.remove(video_path)  # Cleanup after processing
        return jsonify({'output': result, 'confidence': confidence})

    except Exception as e:
        print(f"Error during detection: {str(e)}")  # Print full error details
        return jsonify({'error': str(e)}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, port=3500)

# Deepfake_Detection_System


# Deepfake Detection System

This is a Deepfake Detection System built using a hybrid CNN-RNN model (ResNeXt + LSTM) with super-resolution (Real-ESRGAN) for improved accuracy. The system is designed to detect deepfake videos and images, using advanced machine learning techniques and heatmap visualizations to analyze facial features.

## Features

- **Deepfake Detection:** Detects manipulated videos and images using a hybrid CNN-RNN model.
- **Super-Resolution:** Uses Real-ESRGAN for enhancing the resolution of input images/videos for better detection accuracy.
- **Heatmap Visualization:** Visualizes the attention areas of the model using Grad-CAM for better interpretability.
- **Face Detection:** Preprocessing step to detect faces before running predictions.
- **Video and Image Processing:** Handles both video and image inputs for detection.

## Installation

To get started with the project, clone this repository to your local machine:


### Instructions for Model Paths and Drive Links

Make sure the path mentioned for each model weight (`model/df_model.pt` and `Backend/weights/RealESRGAN_x4.pth`) aligns with the locations in your project.
The links for the deepfake model df_model.pt is `https://drive.google.com/file/d/1Stu-Oc-YJTXtpn-Eb6TRphxTro5Decte/view?usp=sharing`
and RealESRGAN model is `https://drive.google.com/file/d/1Stu-Oc-YJTXtpn-Eb6TRphxTro5Decte/view?usp=drive_link`

Let me know if you need further customization!
```bash
"git clone https://github.com/your-username/Deepfake_Detection_System.git"




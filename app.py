import streamlit as st
import numpy as np
from PIL import Image
import pickle
import torch
import torch.nn as nn

# Constants
NUM_CLASSES = 10
IMG_SIZE = 28
CHANNELS = 1
HIDDEN_DIM = 64
DEVICE = torch.device("cpu")  # Streamlit Community Cloud uses CPU

# Residual Block for Generator
class ResBlockGen(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGen, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            output_padding=1 if stride==2 else 0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=1, stride=stride,
                    output_padding=1 if stride==2 else 0
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# Generator Model
class DigitGenerator(nn.Module):
    def __init__(self):
        super(DigitGenerator, self).__init__()
        self.init_size = IMG_SIZE // 2  # Start at 14x14
        self.l1 = nn.Linear(NUM_CLASSES, HIDDEN_DIM * self.init_size ** 2)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM * self.init_size ** 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_blocks = nn.Sequential(
            ResBlockGen(HIDDEN_DIM, HIDDEN_DIM//2, stride=2),  # Upsample to 28x28
            nn.Conv2d(HIDDEN_DIM//2, CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, labels):
        one_hot = torch.zeros(labels.size(0), NUM_CLASSES, device=DEVICE)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        out = self.l1(one_hot)
        out = self.bn1(out)
        out = self.relu(out)
        out = out.view(out.size(0), HIDDEN_DIM, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    return model

# Generate images
def generate_mnist_images(digit, num_images=5):
    model = load_model()
    # Create tensor of repeated digit labels
    labels = torch.tensor([digit] * num_images, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated_imgs = model(labels)  # Shape: [num_images, 1, 28, 28]
    # Rescale from [-1, 1] to [0, 255]
    generated_imgs = (generated_imgs * 0.5 + 0.5) * 255  # Denormalize
    generated_imgs = generated_imgs.cpu().numpy()  # To NumPy
    # Convert to list of 28x28 arrays (remove channel dimension)
    return [generated_imgs[i, 0, :, :] for i in range(num_images)]

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stTextInput > div > div > input {
        width: 100px;
        margin: 0 auto;
        text-align: center;
        border: 2px solid #3B82F6;
        border-radius: 5px;
        padding: 5px;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        display: block;
        margin: 10px auto;
    }
    .stButton > button:hover {
        background-color: #1E40AF;
    }
    .image-caption {
        text-align: center;
        font-size: 0.9em;
        color: #4B5563;
    }
    .stError {
        text-align: center;
        font-size: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.markdown('<div class="title">MNIST Image Generator</div>', unsafe_allow_html=True)

# Input form
with st.container():
    digit_input = st.text_input("Enter a digit (0-9):", max_chars=1, key="digit_input")
    
    if st.button("Generate Images"):
        if digit_input.isdigit() and 0 <= int(digit_input) <= 9:
            digit = int(digit_input)
            with st.spinner("Generating images..."):
                images = generate_mnist_images(digit, num_images=5)
            
            st.success(f"Generated 5 images for digit {digit}")
            
            st.write("**Generated Images:**")
            cols = st.columns(5)
            for i, img_array in enumerate(images):
                img = Image.fromarray(np.uint8(img_array), mode="L")
                cols[i].image(img, caption=f"Image {i+1}", use_column_width=True)
        else:
            st.error("Please enter a valid digit (0-9).")
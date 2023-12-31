from shutil import copyfile
import matplotlib.pyplot as plt
import re
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset
from PIL import Image
import torch.nn as nn
import re
import streamlit as st
from MobileNetV2CBAM import MobileNetV2
# from CBAM import CBAM
from PIL import ImageOps
"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

# set title of app
st.title("Demo Program Skripsi - Penerapan MobileNetV2 Untuk Klasifikasi Emosi Berdasarkan Ekspresi Wajah Pengemudi")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")

def predict(image):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
#     model_state_dict = torch.load('kmu_best_model_adam_sesuai k-fold.pth')
#     model_state_dict = model_state_dict.to(device)
    model_state_dict = torch.load('kmu_best_model_adam_sesuai k-fold.pth', map_location=torch.device('cpu'))

    # create a ResNet model
    model = MobileNetV2(n_class=6, input_size=224, width_mult=1.)
#     device = torch.device('cpu')
#     model = model.to(device)
    model.load_state_dict(model_state_dict) 
    # transform the input image through resizing, normalization

    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.2, 0.2, 0.2)),
            ])
    # load the image, pre-process it, and make predictions
    img = Image.open(image)
#     img = ImageOps.grayscale(img)
    image = ImageOps.grayscale(img)
    image = image.convert("RGB")
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    
    batch_t = torch.unsqueeze(transform(image), 0)
#     batch_t = batch_t.to(device)
    model.eval()
    out = model(batch_t)
    
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:6]]

    
if file_up is not None:
    # display image that user uploaded

    image = Image.open(file_up)
#     st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
#     image = ImageOps.grayscale(image)
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
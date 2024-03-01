# Load Libraries
import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from torchvision import models
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import copy
from copy import deepcopy
import sys
import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
import time

sys.path.append('../segmentation')
from UNet import UNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # dead kernel for matplotlib

# Load Dataset

metadata = pd.read_csv('../doby_meta.csv')
metadata = metadata[metadata['subject_id'] < 16000000]

ORIG_BASE_PATH = '../physionet.org/files/mimic-cxr-jpg/2.0.0'

TRANSFORMS = transforms.Compose([
    transforms.ToTensor()
])

class Dataset(Dataset):
    def __init__(self, metadata, orig_base_path, transform=None):
        self.metadata = metadata
        self.orig_base_path = Path(orig_base_path)
        self.transform = transform

    def __getitem__(self, idx):
        x_path = self.metadata.loc[idx, 'DicomPath']
        x_orig_path = self.orig_base_path / Path(x_path)
        
        x_orig = Image.open(x_orig_path).convert('L').resize((224, 224))

        y = self.metadata.loc[idx, 'normal']

        if self.transform:
            x = self.transform(x_orig)

        return x, y

    def __len__(self):
        return self.metadata['normal'].count()

ds = Dataset(metadata, ORIG_BASE_PATH, TRANSFORMS)

# Load Models

class customDenseNet(nn.Module):
    def __init__(self):
        super(customDenseNet, self).__init__()
        self.model = models.densenet121()
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.linear = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

segmentation_model = UNet(channel=1)
segmentation_model.load_state_dict(torch.load('../segmentation/parameters/unet_best_f1_model_state.pt', map_location=device))

classification_model = customDenseNet()
classification_model.load_state_dict(torch.load('../classification/densenet_parameters/tune_best_f1_model_state.pt', map_location=device))

classification_model.eval()
segmentation_model.eval()

# Load Inference

transform = transforms.ToTensor()

def inference(X, model1, model2, toTensor):
    with torch.no_grad():
        input = X.unsqueeze(dim=0)
        segmentation_tensor = model1(input)
        segmentation_tensor = segmentation_tensor.squeeze()
        segmentation_img = to_pil_image(segmentation_tensor)
        original_img = to_pil_image(X)

        seg_mask = np.array(segmentation_img)
        seg_mask = np.where(seg_mask > 128, 1, 0)
        lung_part_img = np.array(original_img) * seg_mask
        lung_part_img = np.array(lung_part_img)
        original_img = original_img.filter(ImageFilter.GaussianBlur(1))

        blending_img = Image.blend(original_img, segmentation_img, 0.2)
        blending_img = ImageOps.equalize(blending_img)
        blending_tensor = toTensor(blending_img).unsqueeze(dim=0)

        predicted = model2(blending_tensor)
        return segmentation_img, lung_part_img, original_img, blending_img, predicted.item()

# streamlit Dashboard

st.title('⭐ U-Net+DenseNet121 Dashboard ⭐')

st.header('Project Description', divider='green')
st.markdown('''
이 프로젝트는 울혈성 심부전 환자들이 심인성 폐부종이라는 급성 질환이 찾아올 수 있다는 
사실에 주목하여 환자의 Chest X-Ray 데이터를 통해 **폐부종을 진단하는 모델**을 개발한 프로젝트입니다.
연구 과정에서 '폐부종을 진단하는 모델이니까 ***폐 부분에 대한 집중도를 높이면 어떨까?***'라는 아이디어로
Lung Segmentation 후, 원본 이미지에는 **Gaussian Filtering**을 하여 두 이미지를 Blending한 데이터를 만들어냈습니다.
이러한 데이터로 DenseNet121 모델에서 폐부종 유무를 진단할 수 있도록 Binary Classifcation을 하는 모델을
구축했습니다.
''')

st.header('Choose Data!', divider='green')
    
n_data = 10
data_list_key = []
data_list_key.extend([f'Patient {i}' for i in range(1, n_data+1)])

data_list_value = [
    35,
    42,
    542,
    77,
    73,
    68,
    253,
    753,
    246,
    46
]
data_dict = dict(zip(data_list_key, data_list_value))

model_on_off = None
selected_data = None

left, right = st.columns(2)

with left:
    selected_data = st.selectbox(
    '모델에 넣을 Chest X-ray 샘플 데이터를 골라주세요.',
    data_list_key,
    placeholder='데이터를 선택하세요...'
    )
    
with right:
    if selected_data is not None:
        st.write(f'당신이 선택한 데이터는 [{selected_data}]입니다.')
        x, y = ds[data_dict[selected_data]]
        
        st.image(to_pil_image(x),
                 caption=selected_data,
                 width=300)

st.header('Segmentation', divider='green')
st.write('폐 분할 이미지를 얻기 위해서 **U-Net을 활용하여 마스크 이미지를 얻어냅니다.**')

seg_img, lung_part, orig_img, blend_img, pred = inference(x, segmentation_model, classification_model, transform)

left2, center2, right2 = st.columns(3)

get_mask_checkbox = None

with left2:
    st.image(to_pil_image(x),
             caption=selected_data,
             width=200)
    seg_checkbox = st.checkbox(label='SEGMENTATION', key='btn1')

with center2:
    if seg_checkbox:
        st.image(seg_img,
                 caption='Segmentation',
                 width=200)
        get_mask_checkbox = st.checkbox(label='GET LUNG PART', key='btn2')

with right2:
    if seg_checkbox and get_mask_checkbox:
        st.image(lung_part,
                 caption='Lung Part',
                 width=200)

st.header('Blending & Classification', divider='green')

blend_button = st.button(label='BLEND & CLASSIFICATION', key='btn3')

if (blend_button is False) and get_mask_checkbox:
    left3, right3 = st.columns(2)
    with left3:
        st.image(lung_part,
                 caption='Lung Part',
                 width=300)
    with right3:
        st.image(orig_img,
                 caption='Original Image with Gaussiang Filter',
                 width=300)
elif (blend_button is True) and get_mask_checkbox:
    left4, right4 = st.columns(2)
    with left4:
        st.image(blend_img,
                 caption='Blended Image',
                 width=300)
    with right4:
        # 0.5 이상 Abnormal
        predicted_value = '**Abnormal**' if pred >= 0.5 else '**Normal**'
        real_value = '**Abnormal**' if y >= 0.5 else '**Normal**'
        st.subheader('Result', divider='rainbow')
        st.write(f'모델은 이 환자를 {predicted_value}이라 진단합니다.\n')
        st.write(f'실제로 이 환자는 {real_value} 입니다.')
else:
    st.write('위 Segmentation 작업을 끝내주세요!')
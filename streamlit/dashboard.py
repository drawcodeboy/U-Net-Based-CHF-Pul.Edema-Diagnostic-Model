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
from time import sleep
from stqdm import stqdm

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

st.title('CHF Patient Pul.Edema Diagnosis Model (U-Net+DenseNet121)')

st.header('1️) Project Description', divider='green')
st.markdown('''
이 프로젝트는 <u><b>울혈성 심부전(Congestive Heart Failure)</b></u> 환자들이 <u><b>폐부종(Pulmonary Edema)</b></u>이라는 급성 질환이 찾아올 수 있다는 사실에 주목하여 환자의 Chest X-Ray 데이터를 통해 **폐부종을 진단하는 모델**을 개발한 프로젝트입니다.

''', unsafe_allow_html=True) 

st.markdown('''연구 과정에서 <b>의료진의 진단 과정</b>이라는 프로세스에 주목하여 폐 부분에 대한 집중도를 높이는 아이디어를 기반으로 U-Net 기반 아키텍처를 활용하여 <u><b>Segmentation</b></u>을 수행했습니다. Segmentation을 수행하는 과정에서 <i><b>U-Net, SA U-Net, U-Net++</b></i>를 학습시켜 비교 실험을 통해 가장 성능이 좋은 모델을 선정했습니다. 원본 이미지에는 **Gaussian Filtering**을 하여 두 이미지를 <u><b>Blending</b></u>한 데이터를 만들어냈습니다.''', unsafe_allow_html=True)

st.markdown('''사전 학습된 <i><b>DenseNet121, VGG16</b></i> 두 모델에 데이터 처리 방식에 있어 5가지 Method에 대한 실험을 하여, 폐부종 유무를 진단할 수 있도록 <u><b>Binary Classifcation</b></u>을 하는 모델을 구축했습니다. 해당 연구에 관한 자세한 Method는 블로그, 깃허브, 논문을 참고해주세요 :smile:
> <b>The data utilized for this project</b>
> * <a href="https://physionet.org/content/mimic-cxr-jpg/2.0.0/">MIMIC-CXR-JPG - chest radiographs with structured labels</a>
> * <a href="https://physionet.org/content/chest-x-ray-segmentation/1.0.0/">Chest X-ray Dataset with Lung Segmentation v1.0.0</a>
> * <a href="https://physionet.org/content/mimic-cxr-pe-severity/1.0.1/">Pulmonary Edema Severity Grades Based on MIMIC-CXR v1.0.1</a>''', unsafe_allow_html=True)

st.header('2) Choose Data!', divider='green')
    
n_data = 10
data_list_key = []
data_list_key.extend([f'Patient {i}' for i in range(1, n_data+1)])

data_list_value = [
    35, # Abnormal
    42, # Normal
    845, # Abnormal
    192, # Normal
    73, # Normal
    70, # Abnormal
    254, # Abmormal
    753, # Normal
    530, # Abnormal
    924 # Normal
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
        st.write(f'당신이 선택한 데이터는 <u><b>[{selected_data}]</b></u>입니다.', unsafe_allow_html=True)
        x, y = ds[data_dict[selected_data]]
        
        st.image(to_pil_image(x),
                 caption=selected_data,
                 width=300)

st.header('3) Segmentation!', divider='green')
st.write('폐 분할 이미지를 얻기 위해서 <u><b>U-Net</b></u>을 활용합니다.', unsafe_allow_html=True)

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
        get_mask_checkbox = st.checkbox(label='GET LUNG IMAGE', key='btn2')

with right2:
    if seg_checkbox and get_mask_checkbox:
        st.image(lung_part,
                 caption='Lung Image',
                 width=200)

st.header('4) Blending & Classification!', divider='green')
st.write('폐 분할 이미지와 Gaussian Filtering된 원본 이미지를 Blending하여 <u><b>DenseNet121</b></u>을 통해 진단합니다.', unsafe_allow_html=True)

blend_button = st.button(label='BLEND & CLASSIFICATION', key='btn3')

if (blend_button is False) and get_mask_checkbox:
    left3, right3 = st.columns(2)
    with left3:
        st.image(lung_part,
                 caption='Lung Image',
                 width=300)
    with right3:
        st.image(orig_img,
                 caption='Gaussian Filtered Original Image',
                 width=300)
elif (blend_button is True) and get_mask_checkbox:
    left4, right4 = st.columns(2)
    with left4:
        st.image(blend_img,
                 caption='Blended Image',
                 width=300)
    with right4:
        # 0.5 이상 Abnormal
        predicted_value = 'Abnormal(폐부종)' if pred >= 0.5 else 'Normal(정상)'
        real_value = 'Abnormal' if y >= 0.5 else 'Normal'
        st.subheader('Classification Result', divider='rainbow')
        st.write(f'모델은 <b>{predicted_value}</b>이라 진단합니다.\n', unsafe_allow_html=True)
        if real_value == 'Abnormal':
            st.write(f'<div style="color:red">실제로 이 환자는 <b>폐부종 환자입니다.</b></div>', unsafe_allow_html=True)
        else:
            st.write(f'<div style="color:green">실제로 이 환자는 <b>폐부종 환자가 아닙니다.</b></div>', unsafe_allow_html=True)
else:
    st.write('위 Segmentation 작업을 끝내주세요!')
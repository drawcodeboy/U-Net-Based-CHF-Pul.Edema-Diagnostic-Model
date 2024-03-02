# Exploring Diagnostic Methodology for Pulmonary Edema Diagnosis in Patients with Congestive Heart Failure Using U-Net Based Architecture

### U-Net 기반 아키텍처를 활용한 울혈성 심부전 환자 폐부종 진단 방법론 연구

#### <div align="center"><b>⭐ 한국정보통신학회 JKIICE 논문 심사 중 ⭐</b></div>

<div align="center"><a href="./PAPER/투고%20초안%20최종.pdf">Paper Link</a></div>

## 📄 Project Description

[Doby's Lab (BLOG): U-Net 기반 아키텍처를 활용한 울혈성 심부전 환자 폐부종 진단 방법론 연구]()

- 추후 추가 예정

## 🏃‍♂️ OMS (One-Man Show Project)

## 📁 Dataset

1. <a href="https://physionet.org/content/mimic-cxr-jpg/2.0.0/">MIMIC-CXR-JPG - chest radiographs with structured labels</a>
2. <a href="https://physionet.org/content/chest-x-ray-segmentation/1.0.0/">Chest X-ray Dataset with Lung Segmentation v1.0.0</a>
3. <a href="https://physionet.org/content/mimic-cxr-pe-severity/1.0.1/">Pulmonary Edema Severity Grades Based on MIMIC-CXR v1.0.1</a>

## 💡 Research IDEA and GOAL

- 본 연구의 아이디어 기반이 되었던 뇌종양을 Segmentation하는 연구, <a href="https://ieeexplore.ieee.org/document/9199562">TSTBS</a>에서는 <u><b>의료진의 진단 과정</b></u>에 착안하여 아키텍처를 구성하였습니다. 이에 따라 Chest X-ray를 통해 Congestive Heart Failure(울혈성 심부전) 환자들의 Pulmonary Edema(폐부종) 진단을 할 때, 의료진의 진단 과정에 착안하여 폐 영역에 대한 집중도를 높이고자 Semantic Segmentation을 사용합니다.
- 분할된 폐 영역 이미지 데이터를 활용한 분류 실험 3가지와 그렇지 않은 분류 실험 2가지를 진행하여 성능을 비교한 결과, <u><b>폐 영역에 대해 고려한 실험이 성능이 더 우수하다</b></u>는 사실을 알게 되었습니다.

## 💻 Summary using <code>Streamlit</code>

<code>streamlit</code>을 통해 구현한 웹으로 프로젝트의 전반적인 프로세스를 요약합니다.

![streamlit_gif](./streamlit/oms_streamlit.gif)

## 1️⃣ Segmentation Task

- Lung Segementation을 수행하는 가장 적합한 모델을 찾기 위해 <u><b>U-Net, SA U-Net, U-Net++</b></u> 아키텍처를 학습하여 성능을 비교하였습니다.
- Segmentation Task의 경우에는 PyTorch의 활용도를 높이기 위해서 세 아키텍처 모두 직접 구현하여 사용했습니다.
  1. <a href="">U-Net Repository</a>
  2. <a href="">SA U-Net Repository</a>
  3. <a href="">U-Net++ Repositiory</a>
- Loss function은 Semantic segmentation에서 보편적으로 쓰이는 <b>Dice Loss</b>를 사용했습니다.
  $$Dice\:Loss = \frac{2\times(|A|\cap|B|)}{|A|+|B|}$$
- 종합적인 학습 스펙은 세 모델 전부 아래와 같습니다.

|       Model        |   Accuracy    |   F1-Score    |      AUC      |      MCC      |
| :----------------: | :-----------: | :-----------: | :-----------: | :-----------: |
|    <b>U-Net</b>    | <b>94.67%</b> | <b>0.9808</b> | <b>0.9749</b> | <b>0.9729</b> |
|   SA U-Net (10%)   |    93.98%     |    0.9684     |    0.9695     |    0.9554     |
|   SA U-Net (20%)   |    93.85%     |    0.9660     |    0.9613     |    0.9521     |
|   U-Net++ (fast)   |    94.60%     |    0.9795     |    0.9720     |    0.9711     |
| U-Net++ (accurate) |    94.59%     |    0.9793     |    0.9722     |    0.9708     |

## 2️⃣ Classification Task

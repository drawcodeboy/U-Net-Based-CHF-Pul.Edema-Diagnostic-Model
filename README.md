# Exploring Diagnostic Methodology for Pulmonary Edema Diagnosis in Patients with Congestive Heart Failure Using U-Net Based Architecture

### U-Net 기반 아키텍처를 활용한 울혈성 심부전 환자 폐부종 진단 방법론 연구

#### <div align="center"><b>⭐ 한국정보통신학회 JKIICE 논문 심사 중 ⭐</b></div>

<div align="center"><a href="./PAPER/투고%20초안%20최종.pdf">Paper Link</a></div>

## 📄 Project Description

[Doby's Lab (BLOG): U-Net 기반 아키텍처를 활용한 울혈성 심부전 환자 폐부종 진단 방법론 연구]()

- 추후 추가 예정

## 🏃‍♂️ Motivation, OMS (One-Man Show Project)

> 본 연구 프로젝트 및 논문 투고의 동기는 지난 해 복학하여 돌아온 학기부터 비롯된 불안에서 시작되었습니다.
>
> 생각의 꼬리를 물고 물어 <b><u>'난 연구를 할 수 있는 사람일까?'</u></b>라는 의문이 들었고, 결론적으로 이를 해소할 수 있는 것은 연구의 시작부터 끝까지 직접 해보는 것 밖에 없겠다고 판단했습니다.
>
> 스스로 (1)프로젝트의 아이디어, (2)구현 및 실험, (3)논문 작성까지 모든 프로세스들을 겪어보는 것을 통해 자신에게 증명을 해주고 싶었던 프로젝트입니다.
>
> 그래서 이 프로젝트의 이름을 저는 <b><u>OMS(One-Man Show)</u></b>라는 이름으로 짓고서 프로젝트를 하는 동안 다시 혼자가 되어 힘껏 노를 저었습니다.
>
> 물론, 아직까지 직접 연구실에 들어가서 연구를 배우고, 해본 적은 없기에 제가 함부로 연구를 해봤다고 말하는 것은 오만한 것이라는 생각이 듭니다.
>
> 다만, 이번 프로젝트를 통해 앞으로 더 배워나가야할 것들, 초점을 두어야 할 곳들이 무엇인지 명확히 알았기에 더 겸손한 자세로 헤쳐나가겠습니다.
>
> 긴 글 읽어주셔서 감사합니다. 짧고 길었던 OMS 프로젝트를 마치며.

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
  1. <a href="">U-Net Implementation Repository</a>
  2. <a href="">SA U-Net Implementation Repository</a>
  3. <a href="">U-Net++ Implementation Repositiory</a>
- Loss function은 Semantic segmentation에서 보편적으로 쓰이는 <b>Dice Loss</b>를 사용했습니다.
  $$DiceLoss = \frac{2\times(|A|\cap|B|)}{|A|+|B|}$$
- 종합적인 학습 스펙은 세 모델 전부 아래와 같습니다.

|    Setting    |      Value       |
| :-----------: | :--------------: |
| Loss function | <u>Dice Loss</u> |
|   Optimizer   |   <u>Adam</u>    |
| Learning rate |   <u>1e-4</u>    |
|  Decay step   |     <u>5</u>     |
|  Activation   |  <u>Sigmoid</u>  |
|    Epochs     |    <u>50</u>     |

|        Model        |       Accuracy       |       F1-Score       |         AUC          |         MCC          |
| :-----------------: | :------------------: | :------------------: | :------------------: | :------------------: |
| <u><b>U-Net</b></u> | <u><b>94.67%</b></u> | <u><b>0.9808</b></u> | <u><b>0.9749</b></u> | <u><b>0.9729</b></u> |
|   SA U-Net (10%)    |        93.98%        |        0.9684        |        0.9695        |        0.9554        |
|   SA U-Net (20%)    |        93.85%        |        0.9660        |        0.9613        |        0.9521        |
|   U-Net++ (fast)    |        94.60%        |        0.9795        |        0.9720        |        0.9711        |
| U-Net++ (accurate)  |        94.59%        |        0.9793        |        0.9722        |        0.9708        |

## 2️⃣ 5 Data Processing Methods

## 3️⃣ Classification Task

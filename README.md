# TinyCD-DHJ
### Baseline Model

Baseline 모델인 TinyCD는 적은 파라미터 개수와 빠른 연산 속도가 장점인 Change Detection 모델입니다. Backbone의 경우 EffecientNet_b4의 일부를 활용합니다. Model Development는 Backbone, Encoder, Decoder와 같은 전반적인 구조를 유지하되, 각 구조에 새로운 모듈을 추가하는 방식으로 진행하였습니다.

![baseline](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/a259566a-1d55-446c-a29f-b2ad6b148a85)


### Backbone: EfficientNet_b4
Backbone의 경우 기존 모델과 동일하게 ImageNet 데이터로 pretrained된 EfficientNet_b4를 사용했지만, Layer를 늘려 모델의 이미지 정보 추출 능력을 향상시키고자 하였습니다.

![backbone](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/ea48e565-88bf-446d-b703-a0d4e7a901f1)


### Encoder: ESAMM (Efficient Self-Attention Mixing Module)
기존 TinyCD의 경우 Encoder에서 서로 다른 시점의 두 이미지 정보를 혼합하는 과정이 단순합니다. 채널 기준 Concatenation과 합성곱 연산만으로 Attention Mask를 만들어 냅니다. 이러한 방식은 서로 다른 두 시점의 이미지 특징을 제대로 반영하지 못 할 가능성이 있다고 생각하여, 두 이미지 간의 차이를 고려한 'skip-sub' 방식의 이미지 특징 혼합 방식을 사용한 TMM(Temporal Mixing Module)을 개발하고 활용하였습니다.

혼합된 정보는 Efficient Self-Attention 연산 방식을 거쳐 Decoder로 전달됩니다. Attention 연산 기법은 단순 합성곱 연산에 비해 연산량이 많기 때문에 경량화된 Attention 기법인 Efficient Self-Attention 기법을 활용하였습니다.

![skipsub](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/91309216-ed23-4f6b-9502-65644f09f351)


### Decoder: SMM (Scale Mixing Module)
기존 TinyCD Decoder의 경우 입력으로 채널 수가 1인 Attention Map을 받아 Up-Sampling과 Pixel-wise Multiplication을 통해 Scale 복원을 진행합니다. 해당 방법에서는 서로 다른 Scale 간의 정보가 너무 단순하게 혼합되고, 혼합된 결과로 채널 수가 하나인 Attention Map으로 압축되는 과정에서 정보가 과도하게 축약된다고 생각하였습니다.

이러한 문제점을 해결하기 위해 기존 TinyCD 모델의 픽셀 기반 MLP 구조를 제거함으로써 Decoder로 전달되는 정보의 양을 보존하고, FFC Block을 활용하는 SMM을 개발하였습니다. FFC(Fast Fourier Convolution)의 경우 주기적으로 반복되는 Feature Extraction에 효과적이기 때문에, 위성 영상의 특정 구조물(건물)이나 지형 변화를 탐지하는데에 효율적일 것이라는 직관에 기반하여 도입하였습니다.

![tinycd_dhj](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/0c0657f1-4092-4bba-9a81-b2d7d73cc560)


### Experiments
모델 성능 평가는 Change Detection 분야의 대표적인 Benchmark 데이터 셋인 LEVIR-CD와 WHU-CD를 활용하였습니다.

Table 1의 경우 LEVIR-CD와 WHU-CD 데이터셋을 대상으로 실험한 결과입니다.

![table1](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/17120ecb-33a2-4db7-8768-c028245e5343)


Table 2의 경우 모델 연구 과정에서 진행한 실험의 결과를 정리한 표입니다. 실제 TinyCD 논문의 F1 score 값은 91.05로 기록되어 있으나, GPU를 비롯한 실험 환경 차이로 재현에 실패하였습니다.

![table2](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/e4e76116-1b8c-4410-9165-9dbdd008ce5e)

Table 3는 파라미터 수가 공개된 고성능 Change Detection 모델들과 파라미터 수를 비교한 결과입니다. LightweightCDNet-base가 TinyCD-dhj 보다 우수한 성능을 보이지만, 이를 제외하고는 다른 모델들에 비해 파라미터 수 대비 높은 성능을 보였습니다.

![table3](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/a30201a2-66f1-4b83-9df6-73ae3e48d994)


### Sentinel Hub Pipeline
Real data에 대한 파이프라인을 구축하기 위해 Sentinel Hub API를 활용하였습니다. Django를 활용하여 간단한 데모용 웹페이지를 구축하고, 원하는 지역과 시간을 입력 받아 Change Detection 모델을 동작해볼 수 있도록 구성하였습니다.

![pipeline](https://github.com/dhjzzang/satellite-change-detection/assets/94889715/083b3084-7317-4274-9d00-185b1c74be60)


### References
https://github.com/AndreaCodegoni/Tiny_model_4_CD


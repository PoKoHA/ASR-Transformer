## Model
### 1. 2D-Attention (나중에 추가할 예정)

![Screenshot from 2021-07-24 03-21-26](https://user-images.githubusercontent.com/76771847/126824955-e44245a3-9ae2-402a-aebb-d486b13b0a8b.png)
![module](https://user-images.githubusercontent.com/76771847/126824739-dfc99133-8f25-4afe-b950-ef22076d3b95.png)

### 2. VGGNet Extractor

![c](https://user-images.githubusercontent.com/76771847/126824793-0e5bcfa1-86a5-4705-a08f-426aaffa8961.png)

**실제 코드에서는 VGGExtractor를 먼저 걸치고 positionEncdoing SUM**
## Reference

[speech-transformer]
**https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462506**

[VGGNet Extractor]
https://arxiv.org/pdf/1706.02737

https://ieeexplore.ieee.org/document/8682586

## Code

**https://github.com/sooftware/speech-transformer**

https://github.com/xingchensong/Speech-Transformer-plus-2DAttention

## Dataset

clovacall: https://github.com/clovaai/ClovaCall

Ai_hub(1,000 hours)

## Result
Titian RTX 1개로 약(2일) 학습

`python main.py --epoch 100 --batch-size 8 --warm-steps 750000 --input-dim 161 --max-len 1000 --gpu 0`

![asr-transformer error rate](https://user-images.githubusercontent.com/76771847/127418684-4c784e9e-2347-41c9-ad50-8c9323ccb189.png)

Error Rate 23%로 2015년 model인 LAS model 보다 1% 높게 나왔지만  Transformer 특성 상 Big Data에 더 좋은 performance를 보여주는 성향이 있음
Clovacall Dataset은 Training이 81시간으로 작은 Dataset(50~60epoch 때부터 training loss가 0으로 수렴하여 더이상 학습하지 않는 overfitting이 일어남)

AI_hub(1,000 hours)로 학습 시 더 좋은 performance 기대

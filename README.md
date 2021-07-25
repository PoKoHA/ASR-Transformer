# 미완성 (문제점)
VGGExtractor layer 중 마지막 Linear output_dim 이 length가 계속 바뀌므로

계속 바뀜 ==> 1 iter만 돌아가고 터짐

차후 수정해보기



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
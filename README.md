# Optical Coherence Tomography (OCT)

<br/>

<p align="center">
    <img src="https://github.com/JeremyPai/OCT/blob/master/image/result.PNG">
</p>

<br/>

This repo is a place to store the code which was written while I was in master degree.
It mainly consist of two parts, which are OCT-related codes and DL-related codes.

<br/>

## OCT Source Code:
+ image reconstruction
+ despeckle process
+ algorithm to get attenuation coeffient of each pixel in the images

## Deep Learning Source Code:
+ create model of ResNet and AttentionResNet
+ data preparation flow
+ create confusion matrix
+ use Grad-CAM to visualize possible features
+ t-SNE to achieve dimension reduction
+ three different training flow, including training with keras, training with tf.GradientTape, and transfer learning

<br/>

We also publish a paper related to this research. 

The main feature is that we are the first one to implement AttentionResNet and accomplish wonderful classification result on brain tumors.

<br/>

<p align="center">
    <img src="https://github.com/JeremyPai/OCT/blob/master/image/AttentionResNet.PNG">
</p>

<br/>

## How to Use
Preprocessing
1. set parameters of image_process/config.json
2. run image_process/main.py to get normalized OCT images

Train and Predict
1. set parameters of every classes used in classification_of_normal_glioma_lymphoma.py
2. run classification_of_normal_glioma_lymphoma.py
3. evaluate the results

<br/>

## The BOIL License ([Biomedical Imaging Lab](https://boil.lab.nycu.edu.tw/))
If you use any code in this repo, please kindly cite the following paper and enjoy :)

```
@article{hsu2022differentiation,
  title={Differentiation of primary central nervous system lymphoma from glioblastoma using optical coherence tomography based on attention ResNet},
  author={Hsu, Sanford PC and Hsiao, Tien-Yu and Pai, Li-Chieh and Sun, Chia-Wei},
  journal={Neurophotonics},
  volume={9},
  number={1},
  pages={015005},
  year={2022},
  publisher={SPIE}
}
```

## Related Article
Related paper and master thesis
- [Differentiation of primary central nervous system lymphoma from glioblastoma using optical coherence tomography based on attention ResNet](https://www.spiedigitallibrary.org/journals/neurophotonics/volume-9/issue-1/015005/Differentiation-of-primary-central-nervous-system-lymphoma-from-glioblastoma-using/10.1117/1.NPh.9.1.015005.full?SSO=1)
- [應用注意力模型於光學同調斷層掃描術之腦瘤分類研究](https://hdl.handle.net/11296/8kvps3)

<br/>

## Resource:
+ [Interpretability of Deep Learning Models with Tensorflow 2.0](https://www.sicara.fr/blog-technique/2019-08-28-interpretability-deep-learning-tensorflow)
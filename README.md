# AI_study_notes
我的AI学习笔记(基于pytorch编程）

* 按不同模型整理到不同的文件夹
* 各文件夹中包含 jupyter notebook 笔记，所涉及的图片及论文

目录
========

* [PyTorch基础](#PyTorch基础)
* [Loss Function](#Loss-Function)
* [CNN](#CNN)
* [RNN](#RNN)
* [LSTM](#LSTM)
* [GRU](#GRU)
* [GCNN](#GCNN)
* [Transformer](#Transformer)
* [Vision Transformer(ViT)](#Vision-Transformer(ViT))
* [Masked Auto Encoder(MAE)](#Masked-Auto-Encoder(MAE))
* [ConvMixer](#ConvMixer)
* [ConvNeXt](#ConvNeXt)
* [U-Net](#U-Net)
* [ResNet](#ResNet)
* [VAE](#VAE)
* [Diffusion_Models](#Diffusion-Models)
* [CLIP](#CLIP)
* [【北邮版CS231N】深度学习与数字视频](#【北邮版CS231N】深度学习与数字视频)


## PyTorch基础
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】
  * 01_PyTorch介绍与张量的创建
  * 02_PyTorch张量的运算API(上)
  * 03_PyTorch张量的运算API(下)
  * 04_PyTorch的Dataset与DataLoader详细使用教程
  * 05_深入刨析PyTorch_DataLoader源码
  * 06_PyTorch中搭建分类网络实例
  * 07_深入刨析PyTorch_nnModule源码
  * 08_深入刨析PyTorch的state_dict_parameters_modules源码
  * 09_深入刨析PyTorch的nn_Sequential及ModuleList源码
  * 10_PyTorch_autograd使用教程
  * 11_PyTorch中如何进行向量微分矩阵微分与计算雅可比行列式
  * 12_如何在PyTorch中训练模型
  * 13_详细推导自动微分Forward与Reverse模式
  * 14_保存与加载PyTorch训练的模型和超参数
  * 15_Dropout原理及其源码实现
  * 16_PyTorch中进行卷积残差模块算子融合
  * 52_Excel/Csv文件数据转成PyTorch张量导入模型代码逐行讲解

## Loss Function
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】
  * 55_PyTorch的交叉熵、信息熵、二分类交叉熵、负对数似然、KL散度、余弦相似度的原理与代码讲解
  
## CNN
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 22_PyTorch nn.Conv2d卷积网络使用教程
  * 23_手写并验证滑动相乘实现PyTorch二维卷积
  * 24_手写并验证向量内积实现PyTorch二维卷积
  * 25_手写实现nn.TransposedConv转置卷积
  * 26_手写卷积与转置卷积的代码总结
  * 27_手写实现PyTorch的DilatedConv和GroupConv
  
## RNN
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 29_PyTorch RNN的原理及其手写复现
  
## LSTM
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 30_PyTorch_LSTM和LSTMP的原理及其手写复现

## GRU
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 31_PyTorch_GRU的原理及其手写复现
  
## GCNN
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 32_基于PyTorch的文本分类项目模型与训练代码讲解

## Transformer
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * P_18~P_21: Transformer难点理解与实现
  
## Vision Transformer(ViT)
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 28_Vision Transformer(ViT)模型原理及PyTorch逐行实现
  
## Masked Auto Encoder(MAE)
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 42_Masked_AutoEncoder(MAE)论文导读与模型详细介绍
  * 43_逐行讲解Masked_AutoEncoder(MAE)的PyTorch代码
  
## ConvMixer
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 17_ConvMixer模型原理及其PyTorch逐行实现

## ConvNeXt
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 38_ConvNeXt论文导读与模型精讲
  * 39_ConvNeXt模型代码逐行讲解
  * 40_ConvNeXt分布式训练代码逐行讲解

## U-Net
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 56_U-Net用于图像分割以及人声伴奏分离原理代码讲解

## ResNet
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 41_ResNet模型精讲以及PyTorch复现逐行讲解
  * 51_基于PyTorch_ResNet18的果蔬分类逐行代码讲解

## VAE
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 68_VQVAE预训练模型的论文原理及PyTorch代码逐行讲解
  
## Diffusion Models
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 54_Probabilistic_Diffusion_Model概率扩散模型理论与完整PyTorch代码详细解读
  * 57_Autoregressive_Diffusion_Model自回归扩散模型用于序列预测论文讲解
  * 58_Improved_Diffusion的PyTorch代码逐行深入讲解
  * 62_Score_Diffusion_Model分数扩散模型理论与完整PyTorch代码详细解读
  * 63_必看！概率扩散模型(DDPM)与分数扩散模型(SMLD)的联系与区别
  * 64_扩散模型加速采样算法DDIM论文精讲与PyTorch源码逐行解读
  * 66_Classifier_Guided_Diffusion条件扩散模型论文与PyTorch代码详细解读

## CLIP
* 来自b站up主deep_thoughts 合集【PyTorch源码教程与前沿人工智能算法复现讲解】:
  * 59_基于CLIP_ViT模型搭建相似图像检索系统

* 来自b站up主 迪哥带你学CV 神器CLIP为多模态领域带来了哪些革命？迪哥2小时精讲OpenAI神器—CLIP模型，原理详解+代码复现！:
  * CLIP 模型解读 (HuggingFace Transformers 库 CLIP 演示)

## 【北邮版CS231N】深度学习与数字视频
 * 来自北京邮电大学门爱东2022秋季学期研究生课程 

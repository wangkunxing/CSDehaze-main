# CSDehaze-main

Efficient Ultra-High-Definition Image Dehazing via Convolution-Guided Self-Attention and Feature Compression.



## Dataset Download

Download the training dataset from the following link.

- https://pan.baidu.com/s/1sqJpxvt1-ONqcuG7RLTq0A 密码：vodp

Download the test dataset from the following link.

  4KID and O-HAZE and RESIDE-OTS:

- https://pan.baidu.com/s/1z1ZIrjIHtK-Fui-umjqVdQ?pwd=raqe 提取码: raqe

BeDDE:

- https://drive.usercontent.google.com/uc?id=12p-MY2ZygT5Tl8q0oFxDIUg9B5Jn042-&export=download


## Model Training

Create a new data folderin the project root directory, and create train and test folders under data, and put the downloaded training set and test set into train and test respectively.
   ```bash
   Run train.py
   ```


## Model Inference

Create a new folder hazy in the project root directory and put the data set to be tested into.
   ```bash
   Run test.py
   ```

## Model FPS test

Create a new folder hazy in the project root directory and put the data set to be tested into.
   ```bash
   Run FPS test.py
   ```

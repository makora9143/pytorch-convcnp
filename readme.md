# ConvCNP: Convolutional Conditional Neural Process, in PyTorch

A PyTorch implementation of Convolutional Conditional Neural Process from the 2019 paper ([arXiv](https://arxiv.org/abs/1910.13556), [ICLR2020](https://openreview.net/forum?id=Skey4eBYPS)) by Jonathan Gordon, Wessel P. Bruinsma, Andrew Y. K. Foong, James Requeima, Yann Dubois, and Richard E. Turner.

(The original code is not published as of 2019/12/28)


<img align="center" src= "https://github.com/makora9143/pytorch-convcnp/blob/master/demo_images/anim.gif" height = 400/>



### Table of Contents
- <a href='#installation'>Dependencies</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Demonstration</a>
- <a href='#references'>Reference</a>


## Dependencies

1. Python 3.7+
2. PyTorch 1.3
3. GPyTorch 1.0
4. Numpy 1.16+
5. Scikit-learn 0.21
6. Fastprogress 0.1.21



## Datasets

### 1D Regression
We provide several kernels to generate datasets for syntethic 1D regression:

- EQ Kernel
- Matern-5/2 Kernel
- Periodic Kernel

### 2D Regression
- MNIST
- CIFAR10


## Train

### 1D Regression
```bash
$ python main1d.py --kernel [eq | matern | periodic]
```
### 2D Regression

```bash
$ python main2d.py --dataset [mnist | cifar10] # Highly recommend to run this code in your GPU environment!
```

## Demonstration

### CIFAR10 Prediction

<img align="center" src= "https://github.com/makora9143/pytorch-convcnp/blob/master/demo_images/cifar10-demo.png" height = 180/>

If you want some more demo (1D reg.), please see our jupyter notebooks.




## Reference
- Jonathan Gordon et al. "Convolutional Conditional Neural Processes" [ICLR2020](https://openreview.net/forum?id=Skey4eBYPS) (accepted)
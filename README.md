# Predicting failure progressions of structural materials via deep learning based on void topology

### Introduction
This work introduces a novel method to predict material failure behaviors based on a combination of persistent homology (PH) and deep multimodal learning, which is accepted by [Acta Materialia](https://www.sciencedirect.com/science/article/pii/S1359645423001933).

### Compatibility
We tested the codes with:
  1) Tensorflow-GPU 1.13.1 under Ubuntu 18.04 and Anaconda3 (Python 3.7 or above)
  2) Tensorflow-GPU 1.13.1/Tensorflow 1.13.1 under Windows 10 and Anaconda3 (Python 3.7 or above)


### Requirements
  1) [Anaconda3](https://www.anaconda.com/distribution/#download-section)
  2) [TensorFlow-GPU 1.13.1 or Tensorflow 1.13.1](https://www.tensorflow.org/install/pip)
  3) [HomCloud API](https://homcloud.dev/python-api/)


### Pre-trained and Test Sample
Please download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1_4Yr7Xnl04WTUSb_eVkLYFjq2J141Kl4?usp=sharing) and ensure that you set up our directories as follows:
```
material-failure-prediction (GitHub main folder)
├── Dataset
│   └── Sample
├── DML-Inputs
├── pretrain
└── Result
```


### Citation
Please cite us if you are using our model or dataset in your research works: <br />

[1] Leslie Ching Ow Tiong, Gunjick Lee, Seok Su Sohn and Donghun Kim, "Predicting failure characteristics of structural materials via deep learning based on nondestructive void topology," *Acta Materialia*, vol. 250, pp. 118862, 2023. (See [link](https://www.sciencedirect.com/science/article/pii/S1359645423001933)).

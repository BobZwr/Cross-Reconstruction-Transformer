# Self-Supervised Time Series Representation Learning via Cross Reconstruction Transformer
The official implementation for our TNNLS paper [Self-Supervised Time Series Representation Learning via Cross Reconstruction Transformer](https://arxiv.org/abs/2205.09928).

## Overview of Cross Reconstruction Transformer
<img width="567" alt="image" src="https://github.com/BobZwr/Cross-Reconstruction-Transformer/assets/62683396/8db0886f-1c2c-4a29-af34-126140a0883f">

<img width="1047" alt="image" src="https://github.com/BobZwr/Cross-Reconstruction-Transformer/assets/62683396/4b535ecf-3e85-43ae-baf6-f29686581ddc">

## Getting Started
### Installation
Git clone our repository, and install the required packages with the following command
```
git clone https://github.com/BobZwr/Cross-Reconstruction-Transformer.git
cd Cross-Reconstruction-Transformer
pip install -r requirements.txt
```
We use torch=1.13.0.

## Training and Evaluating
We provide the sample script for training and evaluating our CRT
```
# For Training:
python main.py --ssl True --sl True --load True --seq_len 256 --patch_len 8 --in_dim 9 --n_classes 6
```

```
# For Testing:
python main.py --ssl False --sl False --load False --seq_len 256 --patch_len 8 --in_dim 9 --n_classes 6
```
We also provide a subset of HAR dataset for training and testing.

If you found the codes and datasets are useful, please cite our paper
```
@article{zhang2022cross,
  title={Cross reconstruction transformer for self-supervised time series representation learning},
  author={Zhang, Wenrui and Yang, Ling and Geng, Shijia and Hong, Shenda},
  journal={arXiv preprint arXiv:2205.09928},
  year={2022}
}
```

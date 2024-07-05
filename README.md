# LineRWKV

Code for ["Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention"](https://arxiv.org/abs/2403.17677) paper.

LineRWKV is a method for lossless and lossy compression of hyperspectral images. The compression algorithm is based on predictive coding where a neural network performs prediction of each pixel based on a causal spatial and spectral context, followed by entropy coding of the prediction residual. The neural network predictor processes the image line-by-line using a novel hybrid attentive-recursive operation that combines the representational advantages of Transformers with the linear complexity and recursive implementation of recurrent neural networks. This allows significant savings in memory and computational complexity while reaching state-of-the-art rate-distortion performance.

BibTex reference:
```
@article{valsesia2024linerwkv,
  title={Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention},
  author={Valsesia, Diego and Bianchi, Tiziano and Magli, Enrico},
  journal={arXiv preprint arXiv:2403.17677},
  year={2024}
}
```

## Dataset
Follow the [HyspecNet-11k](https://git.tu-berlin.de/rsim/hsi-compression) repository to prepare the dataset files.


## Model configuration
Model hyperparameters can be changed via the config.py file. The most significant are the following:
```
self.dataset_dir = "/home/valsesia/Scripts/recurrent_space/Dataset/hyspecnet-11k/" # root directory of dataset
self.dataset_difficulty = "hard" # dataset split for hyspecnet-11k
self.dim_enc = 64 # number of feature channels at encoder
self.N_layers_encoder = 2 # encoder layers
self.n_layer_lines = 2 # line predictor layers
self.n_layer_bands = 2 # spectral predictor layers
self.dim_att = 64 # number of feature channels for rwkv attention
self.dim_ffn = 64 # number of feature channels for rwkv ffw
self.ctx_len = 512 # max sequence length for parallel training
self.N_layers_decoder = 2 # decoder layers
self.batch_size = 8  # number of images in minibatch 
self.pos_size = 32 # number of pixels randomly subsampled after encoder to save memory in training
self.epoch_count = 4000 # training epochs
self.lr_init = 1e-4 # initial learning rate
self.lr_final = 1e-6 # final learning rate
```

The hyperparameters used by the various configurations are the following.

|        | **self.dim_enc** | **self.dim_att** | **self.dim_ffn** | **self.N_layers_encoder** | **self.n_layer_lines** | **self.n_layer_bands** | **self.N_layers_decoder** |
|--------|:----------------:|:----------------:|:----------------:|:-------------------------:|:----------------------:|:----------------------:|:-------------------------:|
| **XS** |        32        |        32        |        32        |             1             |            2           |            2           |             1             |
| **S**  |        64        |        64        |        64        |             2             |            2           |            2           |             2             |
| **M**  |        64        |        64        |        64        |             4             |            4           |            4           |             4             |
| **L**  |        96        |        96        |        96        |             4             |            6           |            6           |             4             |


## Training
Launch the training script
```
./launcher_train.sh
```


## Testing/Compressing an image
**Compression tests require image files to be provided as raw files with BSQ ordering, little-endian 16-bit unsigned integers.**
Compress a BSQ raw image with
```
./launcher_compress.sh
```
The quantization step size is set via parameter delta as 2*delta+1. delta=0 performs lossless compression. Paper results use delta=(0 1 2 3 4 5 10 20 40).

The compressed files are the script variables:
- $compressed_file : this contains the entropy-encoded residuals
- $numerical_warning_file : this contains compressed signaling to prevent numerical errors
- $side_info_file_{mu,sigma} : this contains side information about normalizations for decoding

Entropy encoder and decoder are provided as binary files, compiled for Linux x86-64. They are directly called by the compressor.py and decompressor.py files.

## Pretrained models
Pretrained models for the XS,S,M,L configurations are provided. They were trained on the HyspecNet-11k hard split. Final .pth files are under Results/linerwkv/{xs,s,m,l} while checkpoints and dumps of the config object are under log_dir/linerwkv/{xs,s,m,l}.

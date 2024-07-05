#!/bin/bash

# modify this path
rootdir="/home/valsesia/Downloads/linerwkv_git/linerwkv"

model="linerwkv"
datestr="xs"
entropy_decoder_binary="$rootdir/Code/decompressor"

model_file="$rootdir/Results/$model/$datestr/rwkv-final.pth"
config_file="$rootdir/log_dir/$model/$datestr/config.txt"

image="ENMAP01-____L2A-DT0000004981_20221102T031513Z_004_V010110_20221116T123616Z-Y01430270_X03030430-SPECTRAL_IMAGE"
N_rows=128
N_cols=128
N_bands=202

# quantization step size = 2*delta+1
delta=0

gpu=0
device="cuda"

compressed_file="$rootdir/Results/$model/$datestr/"$image"_compressed_$delta.cmp"
residuals_file="$rootdir/Results/$model/$datestr/"$image"_temp_residuals_$delta.bsq"
side_info_file="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5"
reconstructed_file="$rootdir/Results/$model/$datestr/"$image"_reconstructed_$delta.mat"
numerical_warning_file="$rootdir/Results/$model/$datestr/"$image"_numerical_$delta.bin"
side_info_file_mu="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_mu.cmp"
side_info_file_mu_bsq="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_mu.bsq"
side_info_file_sigma="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_sigma.cmp"
side_info_file_sigma_bsq="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_sigma.bsq"

cd $model
CUDA_VISIBLE_DEVICES=$gpu python3 decompressor.py --model_file $model_file --config_file $config_file --output_image $reconstructed_file \
--N_rows $N_rows --N_cols $N_cols --N_bands $N_bands --compressed_file $compressed_file --side_info_file $side_info_file --residuals_file $residuals_file \
--numerical_warning_file $numerical_warning_file --delta_quantization $delta --device $device --entropy_decoder_binary $entropy_decoder_binary
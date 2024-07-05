#!/bin/bash

# modify this path
rootdir="/home/valsesia/Downloads/linerwkv_git/linerwkv"

model="linerwkv"
datestr="xs"
interleave="BSQ"
entropy_encoder_binary="$rootdir/Code/compressor"

model_file="$rootdir/Results/$model/$datestr/rwkv-final.pth"
config_file="$rootdir/log_dir/$model/$datestr/config.txt"

# Input image must be a BSQ file (uint16, little endian)
image="ENMAP01-____L2A-DT0000004981_20221102T031513Z_004_V010110_20221116T123616Z-Y01430270_X03030430-SPECTRAL_IMAGE"
input_image="$rootdir/Dataset/$image.bsq"
N_rows=128
N_cols=128
N_bands=202

# quantization step size = 2*delta+1
delta=0

gpu=0
device="cuda"


echo $input_image
image=$(basename $input_image .bsq)

residuals_file="$rootdir/Results/$model/$datestr/"$image"_residuals_$delta.bsq"
compressed_file="$rootdir/Results/$model/$datestr/"$image"_compressed_$delta.cmp"
side_info_file="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5"
reconstructed_file="$rootdir/Results/$model/$datestr/"$image"_reconstructed_$delta.mat"
numerical_warning_file="$rootdir/Results/$model/$datestr/"$image"_numerical_$delta.bin"
side_info_file_mu="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_mu.cmp"
side_info_file_mu_bsq="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_mu.bsq"
side_info_file_sigma="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_sigma.cmp"
side_info_file_sigma_bsq="$rootdir/Results/$model/$datestr/"$image"_side_info_$delta.h5_sigma.bsq"

cd $model
CUDA_VISIBLE_DEVICES=$gpu python3 compressor.py --model_file $model_file --config_file $config_file --input_image $input_image \
--input_image_interleave $interleave --N_rows $N_rows --N_cols $N_cols --N_bands $N_bands --side_info_file $side_info_file \
--numerical_warning_file $numerical_warning_file --output_file $compressed_file --residuals_file $residuals_file --output_image $reconstructed_file \
--entropy_encoder_binary $entropy_encoder_binary --delta_quantization $delta --device $device


# evaluate total rate
ratecmp=`wc -c $compressed_file | awk '{print $1}'` # need entropy coder to measure this
ratenum=`wc -c $numerical_warning_file | awk '{print $1}'`
rateside0=$(( 64*$N_bands )) 
ratesidemu=`wc -c $side_info_file_mu | awk '{print $1}'`
ratesidesigma=`wc -c $side_info_file_sigma | awk '{print $1}'`
rateside=$(( $rateside0+$ratesidemu+$ratesidesigma ))
rate=$(( $ratecmp+$ratenum+$rateside  ))
echo "$rate bytes"

import os
import subprocess
import argparse
import numpy as np
import json
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Config
from inference import LineRWKV_RNN_spatial, LineRWKV_RNN_spectral_parallel, LineEncoder, LineDecoder, BandZeroDecoder

import scipy.io as sio
import h5py

import torchac

# python implementation of MATLAB multibandread
def multibandread(fname,shape,dtype,offset,interleave):
    
    with open(fname, 'rb') as f:
    
        data = np.zeros(shape,dtype=dtype)

        if interleave == 'BIL':
            for i in range(shape[0]):
                for j in range(shape[2]):
                    f.seek(offset + (i + j*shape[0])*shape[1]*dtype.itemsize)
                    data[i,:,j] = np.fromfile(f, dtype=dtype, count=shape[1])

        if interleave == 'BSQ':
            for i in range(shape[2]):
                f.seek(offset + i*shape[0]*shape[1]*dtype.itemsize)
                data[:,:,i] = np.fromfile(f, dtype=dtype, count=shape[0]*shape[1]).reshape(shape[0],shape[1])

    return data


# python implementation of MATLAB multibandwrite
def multibandwrite(data,fname,interleave):
    
    with open(fname, 'wb') as f:
    
        if interleave == 'BIL':
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    data[i,:,j].tofile(f)

        if interleave == 'BSQ':
            for i in range(data.shape[2]):
                data[:,:,i].tofile(f)
    
    return


# map prediction error to non-negative values
def map_error(e):
    
    em = e+0
    em[e>=0] = e[e>=0]*2
    em[e<0] = np.abs(e[e<0]*2+1)

    return em
        


parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default='', help='pth file with trained model')
parser.add_argument('--config_file', default='', help='config file')
parser.add_argument('--input_image', default='', help='RAW image data (BSQ or BIL)')
parser.add_argument('--input_image_interleave', default='BSQ', help='BSQ or BIL')
parser.add_argument('--N_rows', type=int, default=0, help='N_rows')
parser.add_argument('--N_cols', type=int, default=0, help='N_cols')
parser.add_argument('--N_bands', type=int, default=0, help='N_bands')
parser.add_argument('--output_file', default='', help='Compressed output file')
parser.add_argument('--residuals_file', default='', help='Mapped prediction residuals file')
parser.add_argument('--numerical_warning_file', default='', help='Numerical warning correction file')
parser.add_argument('--entropy_encoder_binary', default='', help='Path to EC binary')
parser.add_argument('--side_info_file', default='', help='Side information H5 file')
parser.add_argument('--output_image', default='', help='Reconstructed MAT image')
parser.add_argument('--delta_quantization', type=int, default=0, help='Half quantization step size')
parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
param = parser.parse_args()


# Import config
config = Config()
with open(param.config_file, 'r') as json_file:
    data_json = json.load(json_file)
    for key, value in data_json.items():
        setattr(config, key, value)
config.model_file = param.model_file
config.device=param.device
param.input_image_interleave = param.input_image_interleave.upper()
config.half = False

# Load data
data = multibandread(param.input_image, (param.N_rows,param.N_cols,param.N_bands), np.dtype(np.uint16), 0, param.input_image_interleave) # (H,W,C) uint16
data = torch.from_numpy(data.astype(np.float32)).float().unsqueeze(0) # (1,H,C,I)
data = data.to(config.device)

# Load model
model_encoder = LineEncoder(config).to(config.device)
model_decoder = LineDecoder(config).to(config.device)
model_bandzero_decoder = BandZeroDecoder(config).to(config.device)
model_spatial_rnn = LineRWKV_RNN_spatial(config).to(config.device)
model_spectral_rnn = LineRWKV_RNN_spectral_parallel(config).to(config.device)
model_spectral_rnn.eval()


data_rec = torch.zeros_like(data).to(config.device) # (1,H,C,I)
with torch.no_grad():

    quant_step_size = (2*param.delta_quantization+1)

    mu_subfactor = 16
    #mu_subfactor = 10
    #mu_subfactor = 1
    mu_enc = torch.round( torch.mean(data[:,::mu_subfactor],dim=(2,),keepdim=True) ) # (1,T,1,I)
    mu =  torch.repeat_interleave( mu_enc / quant_step_size , mu_subfactor, dim=1 )
    sigma_enc = torch.round( torch.std(data[:,::mu_subfactor],dim=(2,),keepdim=True) ) # (1,T,1,I)
    sigma = torch.repeat_interleave( sigma_enc / quant_step_size + 1e-8 , mu_subfactor, dim=1 )
    
    data = torch.round(data/quant_step_size)    
    data = (data-mu)/sigma # (1,T,C,I)

    #print('--Running neural network--')
    quantized_residuals = torch.zeros(data.shape[1],data.shape[2],data.shape[3], device=config.device) # (T,C,I)

    # Encode first line for all bands
    quantized_residuals[0,0,0] = data[0,0,0,0]*sigma[0,0,0,0]+mu[0,0,0,0]
    for band_no in range(data.shape[3]):
        if band_no == 0:
            #dpcm over cols
            first_lines_rec = torch.zeros(data.shape[2],data.shape[3], device=config.device) # (C,I)
            first_lines_rec[0,0] = data[0,0,0,0]*sigma[0,0,0,band_no]+mu[0,0,0,band_no]
            for col_no in range(data.shape[2]-1):
                pred = torch.round(first_lines_rec[col_no,0])
                res = data[0,0,col_no+1,0]*sigma[0,0,0,band_no]+mu[0,0,0,band_no] - first_lines_rec[col_no,0]
                quantized_residuals[0,col_no+1,0] = torch.round(res)
                first_lines_rec[col_no+1,0] = pred + quantized_residuals[0,col_no+1,0]
        else:   
            #dpcm over bands
            pred = torch.round(first_lines_rec[:,band_no-1]) # (C,)
            res = data[0,0,:,band_no]*sigma[0,0,0,band_no]+mu[0,0,0,band_no] - first_lines_rec[:,band_no-1] # (C,)
            quantized_residuals[0,:,band_no] = torch.round(res) # (C,)
            first_lines_rec[:,band_no] = pred + quantized_residuals[0,:,band_no] # (C,)


    # Encode rest of image
    C,I = first_lines_rec.shape
    data_rec[0,0,:,:] = first_lines_rec
    first_lines_rec = (first_lines_rec.unsqueeze(0).unsqueeze(0) - mu[0,0,:,:])/sigma[0,0,:,:] # (1,1,C,I) 
    
    numerical_warning = torch.zeros((data.shape[1]-1,C,I), dtype=torch.int16, device=config.device) # (H-1,C,I)


    for line_no in range(data.shape[1]-1):

        # encode first line
        if line_no == 0:
            a = model_encoder(first_lines_rec)[:,0,:,:,:] # (1,F,C,I)

        # spatial prediction
        x_spatial = model_spatial_rnn(a) # (C,I,F)

        # line-residual for band zero
        a = torch.reshape(a.permute((0,2,1,3)),[C,I,config.dim_enc]) # (C,I,F)
        predicted_value = model_bandzero_decoder(x_spatial[:,:1,:]) # (C,1,1)
        predicted_value = predicted_value[:,0,0] + (data_rec[0,line_no,:,0] - mu[0,line_no,:,0])/sigma[0,line_no,:,0] # (C,)
        predicted_value = predicted_value*sigma[0,line_no+1,:,0]+mu[0,line_no+1,:,0]
        predicted_value_round = torch.round(predicted_value)
        pos_numerical_warning = (torch.abs(predicted_value_round-predicted_value)>(0.5-1e-3)).type(torch.int16)
        numerical_warning[line_no,:,0] = pos_numerical_warning
        predicted_value_round = ((1-pos_numerical_warning)*predicted_value_round + pos_numerical_warning*torch.floor(predicted_value)).type(torch.float32)
        predicted_value_round = torch.clamp(predicted_value_round,-16383,16383)
        quantized_residuals[line_no+1,:,0] = torch.round( data[0,line_no+1,:,0]*sigma[0,line_no+1,:,0]+mu[0,line_no+1,:,0] - predicted_value_round ) # (C,)
        data_rec[0,line_no+1,:,0] = predicted_value_round + quantized_residuals[line_no+1,:,0]

        # encoder
        a_tmp = model_encoder(data[:,(line_no+1):(line_no+2),:,:])[:,0,:,:,:] # (1,F,C,I)
        a = torch.reshape(torch.permute(a_tmp,[0,2,3,1]),[C,I,config.dim_enc]) # (C,I,F)

        # spectral prediction 
        delta = model_spectral_rnn(a, x_spatial) # (C,I,F)

        # decoder
        predicted_value = model_decoder(x_spatial[:,1:,:], delta[:,:-1,:]) # (C,I-1)
        predicted_value = predicted_value*sigma[0,line_no+1,:,1:]+mu[0,line_no+1,:,1:]
        predicted_value_round = torch.round(predicted_value)
        pos_numerical_warning = (torch.abs(predicted_value_round-predicted_value)>(0.5-1e-3)).type(torch.int16)
        numerical_warning[line_no,:,1:] = pos_numerical_warning
        predicted_value_round = ((1-pos_numerical_warning)*predicted_value_round + pos_numerical_warning*torch.floor(predicted_value)).type(torch.float32)
        predicted_value_round = torch.clamp(predicted_value_round,-16383,16383)
        quantized_residuals[line_no+1,:,1:] = torch.round( data[0,line_no+1,:,1:]*sigma[0,line_no+1,:,1:]+mu[0,line_no+1,:,1:] - predicted_value_round ) # (C,I-1)
        data_rec[0,line_no+1,:,1:] = predicted_value_round + quantized_residuals[line_no+1,:,1:]

        a = torch.permute(torch.reshape(a,[1,C,I,config.dim_enc]),[0,3,1,2]) # (C,I,F) --> (1,C,I,F) --> (1,F,C,I)


quantized_residuals = torch.round(quantized_residuals).cpu().numpy().astype(np.int16)
mapped_prediction_error = map_error(quantized_residuals).astype(np.uint16)


# entropy encoding of numerical warning
numerical_warning = numerical_warning.cpu()
cdf = torch.zeros((data.shape[1]-1,C,I,3), dtype=torch.float32)
cdf[:,:,:,1] = 1-1e-3
cdf[:,:,:,2] = 1
byte_stream = torchac.encode_float_cdf(cdf, numerical_warning, check_input_bounds=True, needs_normalization=True)
with open(param.numerical_warning_file, 'wb') as fout:
    fout.write(byte_stream)

# entropy encoding of mu, sigma
mu_enc = mu_enc[0].cpu().numpy().astype(np.int16)
mu_diff = mu_enc[1:] - mu_enc[:-1]
mu_diff_mapped = map_error(mu_diff).astype(np.uint16)
multibandwrite(mu_diff_mapped, param.side_info_file+'_mu.bsq', param.input_image_interleave)
subprocess.run([param.entropy_encoder_binary, "--input", param.side_info_file+'_mu.bsq', "--output", param.side_info_file+'_mu.cmp', "--rows", str(int(mu_diff_mapped.shape[0])), "--columns", "1", "--bands", str(param.N_bands), "--in_format", param.input_image_interleave, "--in_depth", "16", "--in_byte_ordering", "little", "--u_max", "18", "--y_0", "1", "--k", "3", "--y_star", "6"])

sigma_enc = sigma_enc[0].cpu().numpy().astype(np.int16)
sigma_diff = sigma_enc[1:] - sigma_enc[:-1]
sigma_diff_mapped = map_error(sigma_diff).astype(np.uint16)
multibandwrite(sigma_diff_mapped, param.side_info_file+'_sigma.bsq', param.input_image_interleave)
subprocess.run([param.entropy_encoder_binary, "--input", param.side_info_file+'_sigma.bsq', "--output",  param.side_info_file+'_sigma.cmp', "--rows", str(int(sigma_diff_mapped.shape[0])), "--columns", "1", "--bands", str(param.N_bands), "--in_format", param.input_image_interleave, "--in_depth", "16", "--in_byte_ordering", "little", "--u_max", "18", "--y_0", "1", "--k", "3", "--y_star", "6"])

with h5py.File(param.side_info_file, 'w') as f:
    f.create_dataset('mu', data=mu.cpu().numpy())
    f.create_dataset('sigma', data=sigma.cpu().numpy())

#data_rec = torch.round(data_rec*sigma+mu)
data_rec = data_rec*(2*param.delta_quantization+1)
sio.savemat(param.output_image, {'data_rec': data_rec[0].cpu().numpy()})

if param.residuals_file != '':
    #print('--Writing mapped residuals file--')
    multibandwrite(mapped_prediction_error, param.residuals_file, param.input_image_interleave)

# Call entropy encoder executable as subprocess
# Example compressor --input "res.bsq" --output "res.cmp" --rows 127 --columns 128 --bands 202 --in_format BSQ --in_depth 16 --in_byte_ordering little --u_max 18 --y_0 1 --k 3 --y_star 6
if param.output_file != '':
    #print('--Running entropy encoder--')
    subprocess.run([param.entropy_encoder_binary, "--input", param.residuals_file, "--output", param.output_file, "--rows", str(param.N_rows), "--columns", str(param.N_cols), "--bands", str(param.N_bands), "--in_format", param.input_image_interleave, "--in_depth", "16", "--in_byte_ordering", "little", "--u_max", "18", "--y_0", "1", "--k", "3", "--y_star", "6"])

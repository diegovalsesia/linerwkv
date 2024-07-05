import os
import subprocess
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Config
from inference import LineRWKV_RNN_spatial, LineRWKV_RNN_spectral, LineEncoder, LineDecoder, BandZeroDecoder


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
        

def unmap_error(em):

    e = em+0
    e[em%2==0] = em[em%2==0]/2
    e[em%2==1] = -(em[em%2==1]+1)/2

    return e




parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default='', help='pth file with trained model')
parser.add_argument('--config_file', default='', help='config file')
parser.add_argument('--output_image', default='', help='Reconstructed MAT image')
parser.add_argument('--N_rows', type=int, default=0, help='N_rows')
parser.add_argument('--N_cols', type=int, default=0, help='N_cols')
parser.add_argument('--N_bands', type=int, default=0, help='N_bands')
parser.add_argument('--compressed_file', default='', help='Compressed mapped prediction residuals')
parser.add_argument('--side_info_file', default='', help='Side information H5 file')
parser.add_argument('--numerical_warning_file', default='', help='Numerical warning correction file')
parser.add_argument('--delta_quantization', type=int, default=0, help='Half quantization step size')
parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
parser.add_argument('--entropy_decoder_binary', default='', help='Path to ED binary')
parser.add_argument('--residuals_file', default='', help='Temporary file for decoded residuals')
param = parser.parse_args()

# Import config
config = Config()
with open(param.config_file, 'r') as json_file:
    data_json = json.load(json_file)
    for key, value in data_json.items():
        setattr(config, key, value)
config.model_file = param.model_file
config.device = param.device
if param.numerical_warning_file != '':
    numerical=True
    cdf = torch.zeros((param.N_rows-1,param.N_cols,param.N_bands,3), dtype=torch.float32)
    cdf[:,:,:,1] = 1-1e-3
    cdf[:,:,:,2] = 1
    with open(param.numerical_warning_file, 'rb') as fin:
        byte_stream = fin.read()
    numerical_warning = torchac.decode_float_cdf(cdf, byte_stream).to(param.device) # (T-1,C,I)

# Load data
# decode residuals
subprocess.run([param.entropy_decoder_binary, "--input", param.compressed_file, "--output", param.residuals_file, "--rows", str(param.N_rows), "--columns", str(param.N_cols), "--bands", str(param.N_bands), "--in_format", "BSQ", "--in_depth", "16", "--in_byte_ordering", "little", "--u_max", "18", "--y_0", "1", "--k", "3", "--y_star", "6"])
mapped_residuals = multibandread(param.residuals_file, (param.N_rows,param.N_cols,param.N_bands), np.dtype(np.uint16), 0, 'BSQ') # (H,C,I) uint16
mapped_residuals = torch.from_numpy(mapped_residuals.astype(np.float32)).float().unsqueeze(0).to(config.device) # (1,H,C,I)
residuals = unmap_error(mapped_residuals) # (1,H,C,I)

with h5py.File(param.side_info_file, 'r') as f:
    #first_lines = torch.from_numpy( np.array(f['first_lines']).astype(np.float32) ) # (C,I)
    mu = torch.from_numpy( np.array(f['mu']).astype(np.float32) ).to(config.device) # (1,T,1,I)
    sigma = torch.from_numpy( np.array(f['sigma']).astype(np.float32) ).to(config.device) # (1,T,1,I)


model_encoder = LineEncoder(config).to(config.device)
model_decoder = LineDecoder(config).to(config.device)
model_bandzero_decoder = BandZeroDecoder(config).to(config.device)
model_spatial_rnn = LineRWKV_RNN_spatial(config).to(config.device)
model_spectral_rnn = LineRWKV_RNN_spectral(config).to(config.device)


with torch.no_grad():

    # Decode first line for all bands
    data_rec = torch.zeros((param.N_rows,param.N_cols,param.N_bands), device=config.device)
    data_rec[0,0,0] = residuals[0,0,0,0]
    
    residuals[:,0,:,:] = residuals[:,0,:,:] 
    #residuals[:,1:,:,0] = residuals[:,0,:,0] # residuals (1,H,C,I)
    
    for band_no in range(param.N_bands):
        if band_no == 0:
            #dpcm over cols
            first_lines_rec = torch.zeros(param.N_cols,param.N_bands, device=config.device) # (C,I)
            first_lines_rec[0,0] = data_rec[0,0,0]
            for col_no in range(param.N_cols-1):
                pred = torch.round(first_lines_rec[col_no,0])
                first_lines_rec[col_no+1,0] = pred + residuals[0,0,col_no+1,0]
        else:   
            #dpcm over bands
            pred = torch.round(first_lines_rec[:,band_no-1]) # (C,)
            first_lines_rec[:,band_no] = pred + residuals[0,0,:,band_no] # (C,)
    data_rec[0,:,:] = first_lines_rec+0.0 # (C,I)

    # Loop over lines
    #print('--Running neural network--')

    # Decode rest of image
    C,I = first_lines_rec.shape
    first_lines_rec = (first_lines_rec.unsqueeze(0).unsqueeze(0) - mu[0,0,:,:])/sigma[0,0,:,:] # (1,1,C,I)
    next_line_hat = torch.zeros((1,1,C,I), device=config.device) # (1,1,C,I)
    for line_no in range(data_rec.shape[0]-1):
        
        # reset spectral state
        model_spectral_rnn.reset_state()

        # encode first line
        if line_no == 0:
            a = model_encoder(first_lines_rec)[:,0,:,:,:] # (1,F,C,I)
            next_line_hat = first_lines_rec+0
        
        # spatial prediction
        x_spatial = model_spatial_rnn(a) # (C,I,F)
        
        # line-residual for band 0
        a = torch.reshape(a.permute((0,2,1,3)),[C,I,config.dim_enc]) # (C,I,F)
        predicted_value = model_bandzero_decoder(x_spatial[:,0:1,:])
        predicted_value = predicted_value[:,0,0] + next_line_hat[0,0,:,0] # (C,)
        predicted_value = predicted_value*sigma[0,line_no+1,:,0]+mu[0,line_no+1,:,0]
        predicted_value = ((1-numerical_warning[line_no,:,0])*torch.round(predicted_value) + numerical_warning[line_no,:,0]*torch.floor(predicted_value)).type(torch.float32)
        predicted_value = torch.clamp(predicted_value,-16383,16383)

        data_rec[line_no+1,:,0] = predicted_value + residuals[0,line_no+1,:,0]

        next_line_hat[0,0,:,0] = (data_rec[line_no+1,:,0] - mu[0,line_no+1,:,0]) / sigma[0,line_no+1,:,0]
        a_tmp = model_encoder(next_line_hat[:,:,:,0:1])[:,0,:,:,:] # (1,F,C,1)
        a[:,0,:] = torch.reshape(torch.permute(a_tmp,[0,2,3,1]),[C,1,config.dim_enc])[:,0,:] # (C,F)

        # spectral prediction
        for band_no in range(I-1):

            delta = model_spectral_rnn(a[:,band_no,:], x_spatial[:,band_no,:]) # (C,1,F)       

            predicted_value = model_decoder(x_spatial[:,band_no+1:band_no+2,:], delta)  # (C,1) prediction for next line, band_no+1
            predicted_value = predicted_value[:,0]*sigma[0,line_no+1,:,band_no+1]+mu[0,line_no+1,:,band_no+1] # (C,)
            predicted_value = ((1-numerical_warning[line_no,:,band_no+1])*torch.round(predicted_value) + numerical_warning[line_no,:,band_no+1]*torch.floor(predicted_value)).type(torch.float32)
            predicted_value = torch.clamp(predicted_value,-16383,16383)

            data_rec[line_no+1,:,band_no+1] = predicted_value + residuals[0,line_no+1,:,band_no+1]
            
            next_line_hat[0,0,:,band_no+1] = (data_rec[line_no+1,:,band_no+1] - mu[0,line_no+1,:,band_no+1]) / sigma[0,line_no+1,:,band_no+1]               
            a_tmp = model_encoder(next_line_hat[:,:,:,band_no+1:band_no+2])[:,0,:,:,:] # (1,F,C,1)
            a[:,band_no+1,:] = torch.reshape(torch.permute(a_tmp,[0,2,3,1]),[C,1,config.dim_enc])[:,0,:] # (C,F)
    
        a = torch.permute(torch.reshape(a,[1,C,I,config.dim_enc]),[0,3,1,2]) # (C,F) --> (1,C,I,F) --> (1,F,C,I)


quant_step_size = (2*param.delta_quantization+1)
data_rec = data_rec*quant_step_size

sio.savemat(param.output_image, {'data_rec': data_rec.cpu().numpy()})

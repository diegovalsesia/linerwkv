import os, math, gc, importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

#import deepspeed
#from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import scipy.io as sio


class LineEncoder(nn.Module):
    def __init__(self, config):
        super(LineEncoder, self).__init__()
        self.config = config

        self.conv_in = nn.Conv1d(1, config.dim_enc, kernel_size=config.kernel_size, stride=1, padding=config.kernel_size//2)
        self.norm_in = nn.LayerNorm(config.dim_enc)
        
        self.convs = nn.ModuleList([nn.Conv1d(config.dim_enc, config.dim_enc, kernel_size=config.kernel_size, stride=1, padding=config.kernel_size//2) for _ in range(config.N_layers_encoder-1)])
        self.norms = nn.ModuleList([nn.LayerNorm(config.dim_enc) for _ in range(config.N_layers_encoder-1)])

    def forward(self, x):
        (B,T,C,I) = x.shape
        x = torch.permute(x,(0,1,3,2)) # (B,T,I,C)
        x = x.reshape(B*T*I,C).unsqueeze(1) # (BTI,1,C)

        x = self.conv_in(x) # (BTI,F,C)
        x = self.norm_in(x.permute(0,2,1)).permute(0,2,1)
        x = F.gelu(x)
        for i,conv in enumerate(self.convs):
            x = conv(x)
            x = self.norms[i](x.permute(0,2,1)).permute(0,2,1)
            x = F.gelu(x) # (BTI,F,C)
        x = torch.permute(x.reshape(B,T,I,self.config.dim_enc,C), (0,1,3,4,2)) # (B,T,F,C,I)
        return x


class LineDecoder(nn.Module):
    def __init__(self, config):
        super(LineDecoder, self).__init__()
        self.config = config

        self.conv_out = nn.Conv1d(config.dim_enc, 1, kernel_size=1, stride=1, padding=0)
        self.convs = nn.ModuleList([nn.Conv1d(config.dim_enc, config.dim_enc, kernel_size=1, stride=1, padding=0) for _ in range(config.N_layers_decoder-1)])
        self.norms = nn.ModuleList([nn.LayerNorm(config.dim_enc) for _ in range(config.N_layers_decoder-1)])

    def forward(self, x, delta):
        #x: (P(T-1),I-1,F)
        #delta: (P(T-1),I-1,F)
        x = x.permute((0,2,1)) # (P(T-1),F,I-1)
        delta = delta.permute((0,2,1)) # (P(T-1),F,I-1)

        x_out = x+delta
        for i,conv in enumerate(self.convs):
            x_out = conv(x_out)
            x_out = self.norms[i](x_out.permute(0,2,1)).permute(0,2,1)
            x_out = F.gelu(x_out) # (BTI,F,C)
        x_out = self.conv_out(x_out) # (P(T-1),1,I-1)
        
        #x_out = self.conv_out(x) + self.conv_out(delta) # (P(T-1),1,I-1)
        
        return x_out[:,0,:] # (P(T-1),I-1)


class BandZeroDecoder(nn.Module):
    def __init__(self, config):
        super(BandZeroDecoder, self).__init__()
        self.config = config

        self.conv_out = nn.Conv1d(config.dim_enc, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #x: (P(T-1),1,F)
        x = x.permute((0,2,1)) # (P(T-1),F,1)

        x_out = self.conv_out(x) # (P(T-1),1,1)
                
        return x_out


class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.dim_enc)
            for i in range(config.dim_enc):
                ddd[0, 0, i] = i / config.dim_enc
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(config.dim_enc, config.dim_ffn, bias=False)
        self.receptance = nn.Linear(config.dim_enc, config.dim_enc, bias=False)
        self.value = nn.Linear(config.dim_ffn, config.dim_enc, bias=False)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.dim_enc)
            for i in range(config.dim_enc):
                ddd[0, 0, i] = i / config.dim_enc
            
            # fancy time_decay
            decay_speed = torch.ones(config.dim_att)
            for h in range(config.dim_att):
                decay_speed[h] = -5 + 8 * (h / (config.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(config.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(config.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(config.dim_enc, config.dim_att, bias=False)
        self.value = nn.Linear(config.dim_enc, config.dim_att, bias=False)
        self.receptance = nn.Linear(config.dim_enc, config.dim_att, bias=False)
        self.output = nn.Linear(config.dim_att, config.dim_enc, bias=False)
        
    def jit_func(self, x):
        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)
        sr, k, v = self.jit_func(x)
        rwkv = sr * RUN_CUDA(B, T, self.config.dim_att, self.time_decay, self.time_first, k, v)
        return self.output(rwkv)



def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)

T_MAX = 512 # config.ctx_len
wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert B * C % min(C, 32) == 0
        
        w = -torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        ctx.save_for_backward(w, u, k, v, y)
        
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert B * C % min(C, 32) == 0
        w, u, k, v, y = ctx.saved_tensors

        gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        
        wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)

        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)

        return (None, None, None, gw, gu, gk, gv)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.dim_enc)
        self.ln2 = nn.LayerNorm(config.dim_enc)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.dim_enc)

        self.att = RWKV_TimeMix(config, layer_id)
        self.ffn = RWKV_ChannelMix(config, layer_id)
        
        if config.tiny_att_dim > 0 and self.layer_id == config.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(config.dim_enc)
            self.tiny_q = nn.Linear(config.dim_enc, config.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(config.dim_enc, config.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(config.dim_enc, config.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        if config.dropout > 0:
            self.drop0 = nn.Dropout(p = config.dropout)
            self.drop1 = nn.Dropout(p = config.dropout)
        
    def forward(self, x, x_emb=None):

        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)

        if self.config.dropout == 0:
            x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if self.config.tiny_att_dim > 0 and self.layer_id == self.config.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (self.config.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class LineRWKV(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.dim_att = config.dim_att
        self.dim_ffn = config.dim_ffn
        self.tiny_att_layer = config.tiny_att_layer
        self.tiny_att_dim = config.tiny_att_dim
        self.n_layer_lines = config.n_layer_lines
        self.n_layer_bands = config.n_layer_bands
        self.dropout = config.dropout

        assert self.dim_att % 32 == 0
        assert self.dim_ffn % 32 == 0

        # use this when input and output have a fixed and small number of bands, e.g. for segmentation
        #self.encoder = LineEncoder_fixedbands(config)
        #self.decoder = LineDecoder_fixedbands(config)

        # use this when input has a potentially variable or large number of bands, e.g. for hyperspectral prediction
        self.encoder = LineEncoder(config)
        self.decoder = LineDecoder(config)
        self.bandzerodec = BandZeroDecoder(config)

        config.n_layer = self.n_layer_lines
        self.blocks_lines = nn.ModuleList([Block(config, i) for i in range(self.n_layer_lines)])
        config.n_layer = self.n_layer_bands
        self.blocks_bands = nn.ModuleList([Block(config, i) for i in range(self.n_layer_bands)])

        if self.dropout > 0:
            self.drop0_lines = nn.Dropout(p = self.dropout)
            self.drop0_bands = nn.Dropout(p = self.dropout)

    def configure_optimizers(self):
        config = self.config
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if ("time_mix" in n) and (config.layerwise_lr > 0):
                lr_1x.add(n)
            elif ("time_decay" in n) and (config.layerwise_lr > 0):
                lr_2x.add(n)
            elif ("time_faaaa" in n) and (config.layerwise_lr > 0):
                lr_1x.add(n)
            elif ("time_first" in n) and (config.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (config.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.named_parameters()}
        
        if config.layerwise_lr > 0:
           optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if config.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": config.weight_decay, "my_lr_scale": 1.0}]
            #if self.deepspeed_offload:
            #    return DeepSpeedCPUAdam(optim_groups, lr=config.lr_init, betas=config.betas, eps=config.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            #return FusedAdam(optim_groups, lr=config.lr_init, betas=config.betas, eps=config.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
            return torch.optim.AdamW(optim_groups, lr=config.lr_init, betas=config.betas, eps=config.adam_eps, weight_decay=config.weight_decay, amsgrad=False)
        else:
            #if self.deepspeed_offload:
            #    return DeepSpeedCPUAdam(optim_groups, lr=config.lr_init, betas=config.betas, eps=config.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            #return FusedAdam(optim_groups, lr=config.lr_init, betas=config.betas, eps=config.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
            return torch.optim.AdamW(optim_groups, lr=config.lr_init, betas=config.betas, eps=config.adam_eps, weight_decay=0, amsgrad=False)

    #@property
    #def deepspeed_offload(self) -> bool:
    #    strategy = self.trainer.strategy
    #    if isinstance(strategy, DeepSpeedStrategy):
    #        cfg = strategy.config["zero_optimization"]
    #        return cfg.get("offload_optimizer") or cfg.get("offload_param")
    #    return False

    def forward(self, x):
        # x: (B,T,C,I)
        # B: batch size
        # T: number of lines (sequence length)
        # C: number of columns
        # I: number of bands
        config = self.config
        B, T, C, I = x.size()
        assert T <= config.ctx_len, "Cannot forward, model ctx_len is exhausted."

        mu = torch.mean(x,dim=(1,2),keepdim=True) # (B,1,1,I)
        sigma = torch.std(x,dim=(1,2),keepdim=True) # (B,1,1,I)
        x = (x-mu)/(sigma+1e-8) # (B,T,C,I)

        a = self.encoder(x) # (B,T,F,C,I)
        B, T, FF, C, I = a.size()
       
        x0_in = torch.reshape(torch.permute(x[:,:-1,:,0:1],[0,2,1,3]),[B*C,T-1,1]) # (B,T-1,C,1) --> (B,C,T-1,1) --> (BC,T-1,1) 
        a = torch.reshape(torch.permute(a,[0,3,4,1,2]),[B*C,I,T,config.dim_enc]) # (B,T,F,C,I) --> (BC,I,T,F) parallelization over columns,bands 
        # random subsampling of B,C,I pixels to limit memory usage
        # requires 1x1 decoder
        if self.mode == 'train':
            self.pos = torch.randperm(B*C)[:self.config.pos_size]
            a = a[self.pos,:,:,:] # (P,I,T,F)
            P = a.size(0)         
            mu = mu.repeat((1,C,1,1)).reshape([B*C,1,I])[self.pos] # (P,1,I)
            sigma = sigma.repeat((1,C,1,1)).reshape([B*C,1,I])[self.pos] # (P,1,I)                
            x0_in = x0_in[self.pos,:,:] # (P,1,T-1)
        else:          
            P = B*C

        # rnn over lines
        a = torch.reshape(a,[-1,T,config.dim_enc]) # (PI,T,F)
        x_spatial = torch.reshape(a,[-1,T,config.dim_enc]) # (PI,T,F)
        x_spatial_emb = x_spatial
        if config.dropout > 0:
            x_spatial = self.drop0_lines(x_spatial)
        if config.tiny_att_dim > 0:
            for block in self.blocks_lines:
                x_spatial = block(x_spatial, x_spatial_emb)
        else:
            for block in self.blocks_lines:
                x_spatial = block(x_spatial)

        delta = a[:,1:,:] - x_spatial[:,:-1,:] # (PI,T-1,F)
        delta = delta.reshape((P,I,T-1,config.dim_enc)).permute((0,2,1,3)).reshape((P*(T-1),I,config.dim_enc)) # (PI,T-1,F) --> (P(T-1),I,F)  
        x_spatial = x_spatial[:,:-1,:].reshape((P,I,T-1,config.dim_enc)).permute((0,2,1,3)).reshape((P*(T-1),I,config.dim_enc)) # (PI,T-1,F) --> (P(T-1),I,F)       
         
        x0_hat = self.bandzerodec(x_spatial[:,0:1,:]) # (P(T-1),1,1)
        x0_hat = torch.reshape(x0_hat,[P,T-1,1]) # (P,T-1,1)
        x0_hat = x0_hat + x0_in
        if self.mode != 'train':
            x0_hat = torch.permute( torch.reshape(x0_hat,[B,C,T-1,1]), [0,2,1,3] ) # (BC,T-1,I-1) --> (B,T-1,C,1)
            x0_hat = x0_hat*sigma[...,0:1] + mu[...,0:1] # (B,T-1,C,1)
        else:
            x0_hat = x0_hat*sigma[...,0:1] + mu[...,0:1] # (P,T-1,1)

        # rnn over bands
        delta_spectral = delta+0.0 # (P(T-1),I,F)
        delta_spectral_emb = delta_spectral
        if config.dropout > 0:
            delta_spectral = self.drop0_bands(delta_spectral)
        if config.tiny_att_dim > 0:
            for block in self.blocks_bands:
                delta_spectral = block(delta_spectral, delta_spectral_emb)
        else:
            for block in self.blocks_bands:
                delta_spectral = block(delta_spectral) # (P(T-1),I,F)
        
        x = self.decoder(x_spatial[:,1:,:], delta_spectral[:,:-1,:]) # (P(T-1),I,F) --> (P(T-1),I-1)
        x = x.reshape((P,T-1,I-1)) # (P,(T-1),I-1)

        if self.mode != 'train':
            x = torch.permute( torch.reshape(x,[B,C,T-1,I-1]), [0,2,1,3] ) # (BC,T-1,I-1) --> (B,T-1,C,I-1)
            x = x*sigma[...,1:] + mu[...,1:] # (B,T-1,C,I-1)
        else:
            x = x*sigma[...,1:] + mu[...,1:] # (P,T-1,I-1)

        x = torch.cat([x0_hat,x],dim=-1) # (B,T-1,C,I) or (P,T-1,I)

        return x
        

    def training_step(self, batch, batch_idx):      
        self.mode = 'train'
        x, y = batch
              
        logits = self(x)

        B, T, C, I = y.size()
        y = torch.reshape(torch.permute(y,[0,2,1,3]),[-1,T,I])[self.pos,:-1,:]
        
        #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = F.l1_loss(logits, y)

        self.log('train_l1',loss)
        
        return loss

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def validation_step(self, batch, batch_idx):  
        self.mode = 'val'    
        x, y = batch
        #x=x[:,:,:,:4]
        #y=y[:,:,:,:4]
        logits = self(x)

        
        #acc = torch.mean((logits>0.5) == y)
        #self.log('val_accuracy',acc)

        l1_val = F.l1_loss(logits, y[:,:-1,:,:])
        self.log('val_l1',l1_val)

        #if batch_idx==0:
            #self.logger.experiment.add_image('GT image',y[9:10,:,:,0])
            #self.logger.experiment.add_image('Predicted image',logits[9:10,:,:,0])
            #self.logger.experiment.add_image('Prediction error',torch.abs(logits[9:10,:,:,0]-y[9:10,:,:,0]))
            #sio.savemat("debug.mat",{'y':y.detach().cpu().numpy(),'y_hat':logits.detach().cpu().numpy()})
        

    def generate_init_weight(self):
        print(
            f"""
############################################################################
# Init model weight...
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.5)
                else:
                    m[n] = p
            else:
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                zero = [".att.key.", ".att.receptance.", ".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                for kk in zero:
                    if kk in n:
                        scale = 0
                if n == "head.weight":
                    scale = 0.5
                if "head_k." in n:
                    scale = 0.1
                if "head_q." in n:
                    scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                m[n] = torch.empty((shape[0], shape[1]), device=self.config.device)

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()

        gc.collect()
        torch.cuda.empty_cache()
        return m
    



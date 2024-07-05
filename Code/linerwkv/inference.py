import os
import gc
import types
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import scipy.io as sio

from torch.utils.cpp_extension import load

MyModule = nn.Module
def __nop(ob):
    return ob
MyFunction = __nop

# JIT compilation for speedup, can be buggy
#if os.environ["RWKV_JIT_ON"] == "1":
#MyModule = torch.jit.ScriptModule
#MyFunction = torch.jit.script_method

DEBUG_TIME=False

class LineEncoder(nn.Module):
    def __init__(self, config):
        super(LineEncoder, self).__init__()
        self.config = config

        self.conv_in = nn.Conv1d(1, config.dim_enc, kernel_size=config.kernel_size, stride=1, padding=config.kernel_size//2)
        self.norm_in = nn.LayerNorm(config.dim_enc)
        
        self.convs = nn.ModuleList([nn.Conv1d(config.dim_enc, config.dim_enc, kernel_size=config.kernel_size, stride=1, padding=config.kernel_size//2) for _ in range(config.N_layers_encoder-1)])
        self.norms = nn.ModuleList([nn.LayerNorm(config.dim_enc) for _ in range(config.N_layers_encoder-1)])

        with torch.no_grad():
            w = torch.load(config.model_file, map_location='cpu')
            w_encoder = {}
            for key in w.keys():
                if key.startswith('encoder.'):
                    w_encoder[key[len('encoder.'):]] = w[key]
            self.load_state_dict(w_encoder)
            self.to(config.device)

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

        with torch.no_grad():
            w = torch.load(config.model_file, map_location='cpu')
            w_decoder = {}
            for key in w.keys():
                if key.startswith('decoder.'):
                    w_decoder[key[len('decoder.'):]] = w[key]
            self.load_state_dict(w_decoder)
            self.to(config.device)

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
                
        return x_out[:,0,:] # (P(T-1),I-1)
    

class BandZeroDecoder(nn.Module):
    def __init__(self, config):
        super(BandZeroDecoder, self).__init__()
        self.config = config

        self.conv_out = nn.Conv1d(config.dim_enc, 1, kernel_size=1, stride=1, padding=0)

        with torch.no_grad():
            w = torch.load(config.model_file, map_location='cpu')
            w_bandzerodec = {}
            for key in w.keys():
                if key.startswith('bandzerodec.'):
                    w_bandzerodec[key[len('bandzerodec.'):]] = w[key]
            self.load_state_dict(w_bandzerodec)
            self.to(config.device)

    def forward(self, x):
        #x: (P(T-1),1,F)
        x = x.permute((0,2,1)) # (P(T-1),F,1)

        x_out = self.conv_out(x) # (P(T-1),1,1)
                
        return x_out


class LineRWKV_RNN_spatial(MyModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.RUN_DEVICE = config.device
        self.state_lines = None

        with torch.no_grad():
            w = torch.load(config.model_file, map_location='cpu')

            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(config.ctx_len+1, -1)[:-1,:]
            
            keys = list(w.keys())

            print_need_newline = False
            for x in keys:
                                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                    if DEBUG_TIME:
                        print(x, w[x].numpy())
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].float()

                w[x].requires_grad = False

                if config.device == 'cuda' and x != 'emb.weight':
                    w[x] = w[x].cuda()

                # if ('blocks' not in x) or ('blocks' in x):
                #     if print_need_newline:
                #         print('\n', end = '')
                #         print_need_newline = False
                #     print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
                # else:
                #     print_need_newline = True
                #     print('.', end = '', flush = True)
     

        # store weights in self.w
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()



    def LN(self, x, w):
        return F.layer_norm(x, (self.config.dim_enc,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = torch.sigmoid(xr @ rw.T)
        k = torch.square(torch.relu(xk @ kw.T))
        kv = k @ vw.T

        return r * kv

    @MyFunction
    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x

        r = torch.sigmoid(xr @ rw.T)
        k = xk @ kw.T
        v = xv @ vw.T

        kk = k
        vv = v
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = a / b
        
        return (r * wkv) @ ow.T 


    def forward(self, a):
        # a: (B,F,C,I) encoding of reconstructed line l-1 with all bands to be used for line-based prediction
        # B: batch size
        # 1: one current line 
        # C: number of columns
        # I: number of bands
        # output: (BC,I,F) spatial prediction of line l for all bands
        with torch.no_grad():
            w = self.w
            config = self.config
     
            B, FF, C, I = a.size()
            a = torch.reshape(torch.permute(a,[0,2,3,1]),[B*C*I,config.dim_enc]) # (B,F,C,I) --> (BCI,F) parallelization over columns,bands

            if self.RUN_DEVICE == 'cuda':
                a = a.cuda()

            # rnn over lines
            x_spatial = a+0.0

            if self.state_lines == None:
                self.state_lines = torch.zeros(config.n_layer_lines * 5, B*C*I, config.dim_enc, device=self.RUN_DEVICE)
                for i in range(config.n_layer_lines):
                    self.state_lines[5*i+4] -= 1e30

            for i in range(config.n_layer_lines):
                if i == 0:
                    x_spatial = self.LN(x_spatial, w.blocks_lines[i].ln0)
                
                ww = w.blocks_lines[i].att
                x_spatial = x_spatial + self.SA(self.LN(x_spatial, w.blocks_lines[i].ln1), self.state_lines, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks_lines[i].ffn
                x_spatial = x_spatial + self.FF(self.LN(x_spatial, w.blocks_lines[i].ln2), self.state_lines, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
            x_spatial = torch.reshape(x_spatial,[B*C,I,config.dim_enc]) # (BCI,F) --> (BC,I,F) 

            return x_spatial



class LineRWKV_RNN_spectral(MyModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.RUN_DEVICE = config.device
        self.state_bands = None

        with torch.no_grad():
            w = torch.load(config.model_file, map_location='cpu')

            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(config.ctx_len+1, -1)[:-1,:]
            
            keys = list(w.keys())

            print_need_newline = False
            for x in keys:
                                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                    if DEBUG_TIME:
                        print(x, w[x].numpy())
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].float()

                w[x].requires_grad = False
                
                if config.device == 'cuda' and x != 'emb.weight':
                    w[x] = w[x].cuda()

                # if ('blocks' not in x) or ('blocks' in x):
                #     if print_need_newline:
                #         print('\n', end = '')
                #         print_need_newline = False
                #     print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
                # else:
                #     print_need_newline = True
                #     print('.', end = '', flush = True)
     

        # store weights in self.w
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()


    def LN(self, x, w):
        return F.layer_norm(x, (self.config.dim_enc,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = torch.sigmoid(xr @ rw.T)
        k = torch.square(torch.relu(xk @ kw.T))
        kv = k @ vw.T

        return r * kv

    @MyFunction
    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x

        r = torch.sigmoid(xr @ rw.T)
        k = xk @ kw.T
        v = xv @ vw.T

        kk = k
        vv = v
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = a / b
        
        return (r * wkv) @ ow.T 
    
    def reset_state(self):
        self.state_bands = None

    def forward(self, a, x_spatial):
        # a: (BC,F) encoding of reconstructed line l-1 for current band
        # x_spatial: (BC,F) spatial prediction for current band
        # B: batch size
        # C: number of columns
        # output: (BC,1,F) decodable prediction of line l for current band
        with torch.no_grad():
            w = self.w
            config = self.config
     
            BC = a.shape[0]

            delta = a - x_spatial # (BC,F)
            
            if self.state_bands == None:
                self.state_bands = torch.zeros(config.n_layer_bands * 5, BC, config.dim_enc, device=self.RUN_DEVICE)
                for i in range(config.n_layer_bands):
                    self.state_bands[5*i+4] -= 1e30

            for i in range(config.n_layer_bands):
                if i == 0:
                    delta = self.LN(delta, w.blocks_bands[i].ln0)
                
                ww = w.blocks_bands[i].att
                delta = delta + self.SA(self.LN(delta, w.blocks_bands[i].ln1), self.state_bands, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks_bands[i].ffn
                delta = delta + self.FF(self.LN(delta, w.blocks_bands[i].ln2), self.state_bands, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
            delta = torch.reshape(delta,[BC,1,config.dim_enc]) # (BC,F) --> (BC,1,F)

            return delta


###### only for lossless or prequantized lossy ######
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



class LineRWKV_RNN_spectral_parallel(MyModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        config.n_layer = config.n_layer_bands
        self.blocks_bands = nn.ModuleList([Block(config, i) for i in range(config.n_layer_bands)])

        with torch.no_grad():
            w = torch.load(config.model_file, map_location='cpu')
            w_blocks = {}
            for key in w.keys():
                if key.startswith('blocks_bands.'):
                    w_blocks[key] = w[key]
            self.load_state_dict(w_blocks)
            self.to(config.device)

    def forward(self, a, x_spatial):

        delta = a - x_spatial # (C,I,F)
        
        for block in self.blocks_bands:
            delta = block(delta)

        return delta


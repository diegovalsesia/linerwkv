class Config(object):

    def __init__(self):

        # trining data
        # hyspecnet-11k
        self.dataset_dir = "/home/valsesia/Scripts/recurrent_space/Dataset/hyspecnet-11k/"
        self.dataset_difficulty = "hard" # "easy"

        
        self.device = "cuda" # "cpu"

        # architecture
        self.dim_enc = 64
        self.dropout = 0
        # line encoder
        self.kernel_size = 3
        self.N_layers_encoder = 2
        # rwkv
        self.n_layer_lines = 2
        self.n_layer_bands = 2
        self.ctx_len = 512 # max sequence length for parallel training
        self.dim_att = 64
        self.dim_ffn = 64
        self.tiny_att_layer = -1
        self.tiny_att_dim = 0    
        # line decoder
        self.N_layers_decoder = 2
        self.residual = False

        # learning
        self.batch_size = 8
        self.pos_size = 16 # subsampling
        self.epoch_count = 4000
        self.layerwise_lr = 1
        self.weight_decay = 0
        self.weight_decay_final = 0
        self.lr_init = 5e-4
        self.lr_final = 1e-6
        self.warmup_steps = 100
        self.betas = (0.9, 0.99)
        self.adam_eps = 1e-8
        self.epoch_save = 10
        self.channels_subset = None


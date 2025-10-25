import argparse
import os
import torch

class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.initialize()

    def initialize(self):
        self.name = "people"
        self.gpu_ids = "0"
        self.checkpoints_dir = "./noise_layers/simswap/checkpoints"
        self.model = "pix2pixHD"
        self.norm = "batch"
        self.use_dropout = False
        self.data_type = 32
        self.verbose = False
        self.fp16 = False
        self.local_rank = 0
        self.isTrain = True
        self.batchSize = 8
        self.loadSize = 1024
        self.fineSize = 512
        self.label_nc = 0
        self.input_nc = 3
        self.output_nc = 3
        self.dataroot = "./datasets/cityscapes/"
        self.resize_or_crop = "scale_width"
        self.serial_batches = False
        self.no_flip = False
        self.nThreads = 2
        self.max_dataset_size = float("inf")
        self.display_winsize = 512
        self.tf_log = False
        self.netG = "global"
        self.latent_size = 512
        self.ngf = 64
        self.n_downsample_global = 3
        self.n_blocks_global = 6
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        self.niter_fix_global = 0
        self.no_instance = False
        self.instance_feat = False
        self.label_feat = False
        self.feat_num = 3
        self.load_features = False
        self.n_downsample_E = 4
        self.nef = 16
        self.n_clusters = 10
        self.image_size = 224
        self.norm_G = "spectralspadesyncbatch3x3"
        self.semantic_nc = 3

        self.initialized = True

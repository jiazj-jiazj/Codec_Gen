import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.vq_layer_ema import VQEmbeddingEMA_DDP
from .layers.vq_layer_predictive_ema import VQEmbeddingEMA_Predictive_DDP
from .layers.vq_layer_gumbel import GumbelVectorQuantizer, GumbelVectorQuantizer_Parallel
from .layers.vq_layer_predictive_gumbel import GumbelVectorQuantizer_Predictive, GumbelVectorQuantizer_Predictive_Parallel
from tfnet.utils.tools import *
from tfnet.model_type import *
    
class MultiFrmVQBottleNeck(nn.Module):
    def __init__(self, feat_dim, bitrate, sampling_rate=16000, config=None, is_enhance_layer=False):
        super(MultiFrmVQBottleNeck, self).__init__()

        self.bitrate_dict = {'0.256k':256,'0.5k':500, '0.512k':512,'1k':1000,'2k':2000,'3k':3000,'6k':6000,'9k':9000,'12k':12000,'24k':24000}
        self.hop_dur = config['vq_in_dur'] #config["dft_size"] * config["hop_vqvae"] / config["sampling_rate"]
        self.bitrate = bitrate
        bit_per_frame = int(self.hop_dur * self.bitrate_dict[bitrate])        
        
        self.config = config
        self.add_packet_loss = config['add_packet_loss']
        self.plc_unit_type = config['plc_unit_type']                   
        
        self.feat_dim = feat_dim 
        self.combine_frames = config["combineVQ_frames"] 

        #vectorQuantize layer
        self.mask_chnls = 0
        if sampling_rate == 16000:
            self.init_codebook_size_16khz(config, bit_per_frame)
        elif sampling_rate == 24000:
            self.init_codebook_size_24khz(config, bit_per_frame)
        elif sampling_rate == 48000:  
            self.init_codebook_size_48khz(config, bit_per_frame, is_enhance_layer)        
        
        self.codebook_dim = self.latent_dim * self.combine_frames // self.codebook_num
        self.conv1x1_1 = nn.Conv2d(self.feat_dim, self.latent_dim, kernel_size=1, stride=1,bias=False)
        nn.init.xavier_normal_(self.conv1x1_1.weight)
        if is_bitrate_scalable(config):
            self.bitrate_list = config["bitrates_selected"]
            self.valid_groups_dict = {'3k': 8, '6k': 16, '9k': 24, '12k': 32}
            conv1x1_2 = []
            for _ in config['bitrates_selected']:
                conv = nn.Conv2d(self.latent_dim+self.mask_chnls, self.feat_dim, kernel_size=1, stride=1,bias=False)
                nn.init.xavier_normal_(conv.weight)
                conv1x1_2.append(conv)
            self.conv1x1_2 = nn.ModuleList(conv1x1_2)
        else:
            self.conv1x1_2 = nn.Conv2d(self.latent_dim+self.mask_chnls, self.feat_dim, kernel_size=1, stride=1, bias=False)
            nn.init.xavier_normal_(self.conv1x1_2.weight)
        
        # if config["use_learned_reshape_bottleneck"]:
            # self.convreshape_1 = nn.Conv2d(self.latent_dim, self.latent_dim*self.combine_frames, kernel_size=self.combine_frames, stride=1,bias=False)
            # nn.init.xavier_normal_(self.convreshape_1.weight)
            # self.convreshape_2 = nn.ConvTranspose2d(self.latent_dim*self.combine_frames+self.mask_chnls, self.latent_dim, kernel_size=self.combine_frames, stride=1, bias=False)
            # nn.init.xavier_normal_(self.convreshape_2.weight)        
        
        if config["use_parallelvq"]:
            if config["use_predictive"]:
                self.vq_layer = GumbelVectorQuantizer_Predictive_Parallel(config, input_dim=self.latent_dim * self.combine_frames, n_embeddings=self.codebook_size[0], groups=self.codebook_num)
            else:
                self.num_its_para = config['num_parallel_vq']
                self.vq_layer_list = nn.ModuleList(
                    GumbelVectorQuantizer_Parallel(config, input_dim=self.latent_dim * self.combine_frames // self.num_its_para, n_embeddings=self.codebook_size[0], groups=self.codebook_num // self.num_its_para) for i in range(self.num_its_para))
        else:
            if config["vq_type"] == 'EMA':                                    
                if config["use_predictive"]:
                    self.vq_layer_list = nn.ModuleList(VQEmbeddingEMA_Predictive_DDP(i, self.codebook_dim, decay=config["decay"], config=config) for i in self.codebook_size)
                else:
                    self.vq_layer_list = nn.ModuleList(VQEmbeddingEMA_DDP(i, self.codebook_dim, decay=config["decay"], config=config) for i in self.codebook_size)
            elif config["vq_type"] == 'Gumbel':
                if config["use_predictive"]:
                    self.vq_layer_list = nn.ModuleList(
                        GumbelVectorQuantizer_Predictive(config, input_dim=self.codebook_dim, n_embeddings=i, groups=config["groups"], combine_groups=config["combine_groups"]) for i in self.codebook_size) 
                else:
                    self.vq_layer_list = nn.ModuleList(
                        GumbelVectorQuantizer(config, input_dim=self.codebook_dim, n_embeddings=i, groups=config["groups"], combine_groups=config["combine_groups"]) for i in self.codebook_size)  
                        
        if config["use_entropy_loss"]:
            self.set_network_entropy_target(self.bitrate_dict[bitrate], config["entropy_fuzz"], self.hop_dur*self.combine_frames)
       
    def init_codebook_size_16khz(self, config, bit_per_frame):
        bit_per_packet = bit_per_frame * config["combineVQ_frames"]
        if self.bitrate == '0.256k': 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1
            if (bit_per_frame < 5) and (bit_per_frame > 1):  # 10ms  2.5bit                    
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 80 if config["use_complete_latent"] else 160
                        self.codebook_num = 1                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 80 if config["use_complete_latent"] else 160
                        self.codebook_num = 2                        
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 80 if config["use_complete_latent"] else 160
                        self.codebook_num = 1
                        self.codebook_size = [32 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 80 if config["use_complete_latent"] else 160
                        self.codebook_num = 1                        
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number
        elif self.bitrate == '0.5k':
            if bit_per_frame == 10: #20ms 10bit
                assert config["combineVQ_frames"] == 1, "Currently only support combineVQ_frames=1 for 0.5k-20ms vq!"
                if config["vq_rate"] == 'complete':
                    self.latent_dim = config["latent_dim"] if "latent_dim" in config.keys() else 640
                    self.codebook_num = 1
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 160 if config["use_complete_latent"] else 640
                    self.codebook_num = 2
                    self.codebook_size = [64 for i in range(self.codebook_num)]  # codeword number

        elif self.bitrate == '0.512k': 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1           
            if bit_per_frame == 5:  # 10ms  5bit                    
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 80 if config["use_complete_latent"] else 160
                        self.codebook_num = 2                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 81 if config["use_complete_latent"] else 162
                        self.codebook_num = 3                        
                        self.codebook_size = [512 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 80 if config["use_complete_latent"] else 160
                        self.codebook_num = 1
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 80 if config["use_complete_latent"] else 160
                        self.codebook_num = 2                        
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number    
        elif self.bitrate == '1k': 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_frame == 5:  # hop5ms  5bit           
                if config["combineVQ_frames"]==8:                        
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 84 if config["use_complete_latent"] else 240
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 84 if config["use_complete_latent"] else 240
                        self.codebook_num = 6                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                elif config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 84 if config["use_complete_latent"] else 240
                        self.codebook_num = 2                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 84 if config["use_complete_latent"] else 240
                        self.codebook_num = 3                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
            elif bit_per_packet == 40:  # hop10ms/10bit/4frm or hop20ms/20bit/2frm  
                if config["vq_rate"] == 'complete': 
                    self.latent_dim = 168 if config["use_complete_latent"] else 288
                    self.codebook_num = 4                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 162 if config["use_complete_latent"] else 288
                    self.codebook_num = 6                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
            elif bit_per_packet == 20:  # hop10ms/10bit/2frm or hop20ms/20bit/1frm             
                if config["vq_rate"] == 'complete': 
                    self.latent_dim = 162 if config["use_complete_latent"] else 288
                    self.codebook_num = 2
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 162 if config["use_complete_latent"] else 288
                    self.codebook_num = 3                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number             
        elif self.bitrate == '3k': 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_frame == 15:  # 5ms  15bit
                # 2x(2^8x64) = 3.2 kbps                    
                if config["combineVQ_frames"]==8:                        
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 84 if config["use_complete_latent"] else 240
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 80 if config["use_complete_latent"] else 240
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                elif config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 84 if config["use_complete_latent"] else 240
                        self.codebook_num = 6                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 80 if config["use_complete_latent"] else 240
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 84 if config["use_complete_latent"] else 192
                        self.codebook_num = 3
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 80 if config["use_complete_latent"] else 192
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 84 if config["use_complete_latent"] else 128
                        self.codebook_num = 2
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 84 if config["use_complete_latent"] else 128
                        self.codebook_num = 3                        
                        self.codebook_size = [128 for i in range(self.codebook_num)]  # codeword number
            elif bit_per_frame == 30:  # 10ms  30bit                    
                if config["combineVQ_frames"]==4:
                    # if config["use_predictive"] and config["use_compressed_channels"]:
                        # self.latent_dim = 80
                        # self.codebook_num = 16                        
                        # self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                    # else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 168 if config["use_complete_latent"] else 288
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 288 
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    # if config["use_predictive"] and config["use_compressed_channels"]:
                        # self.latent_dim = 80
                        # self.codebook_num = 8                        
                        # self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                    # else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 162 if config["use_complete_latent"] else 240
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 240
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    # if config["use_predictive"] and config["use_compressed_channels"]:
                        # self.latent_dim = 80
                        # self.codebook_num = 4                        
                        # self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                    # else:
                    # if config['use_subcodebook_residue']:
                        # if config["vq_rate"] == 'complete': 
                            # self.latent_dim = 480 if config["use_complete_latent"] else 192
                            # self.codebook_num = 3
                            # self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                        # else:
                            # self.latent_dim = 640 if config["use_complete_latent"] else 256
                            # self.codebook_num = 4                        
                            # self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    # else:                            
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 162 if config["use_complete_latent"] else 192
                        self.codebook_num = 3
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 192
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                                       
        elif self.bitrate == '6k':
            if bit_per_frame == 120: #20ms 120bit
                assert config["combineVQ_frames"] == 1, "Currently only support combineVQ_frames=1 for 6k-20ms vq!"
                if config["vq_rate"] == 'complete':
                    self.latent_dim = 168 if config["use_complete_latent"] else 1440
                    self.codebook_num = 12
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 160 if config["use_complete_latent"] else 1440
                    self.codebook_num = 16
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
            elif bit_per_frame == 60:   # 10ms  60bit
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 168 if config["use_complete_latent"] else 288
                        self.codebook_num = 24                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 288
                        self.codebook_num = 32                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 168 if config["use_complete_latent"] else 240
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 240
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 162 if config["use_complete_latent"] else 192
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 192
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                    self.mask_chnls = 1 
            elif bit_per_frame == 30:  # 5ms  30bit  
                if config["combineVQ_frames"]==8:  # 240bit                    
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 96 if config["use_complete_latent"] else 360                            
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 96 if config["use_complete_latent"] else 384
                        self.codebook_num = 32
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1             
                elif config["combineVQ_frames"]==4:  # 120bit                    
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 84 if config["use_complete_latent"] else 360                            
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 80 if config["use_complete_latent"] else 480 #320 #480
                        self.codebook_num = 16
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1                                             
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 84 if config["use_complete_latent"] else 240                            
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 80 if config["use_complete_latent"] else 240
                        self.codebook_num = 8
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                     
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1
                else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 84 if config["use_complete_latent"] else 120
                        self.codebook_num = 3
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 80 if config["use_complete_latent"] else 120
                        self.codebook_num = 4
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                    
                    if self.add_packet_loss:
                        self.mask_chnls = self.latent_dim if self.plc_unit_type=='full' else (1 if self.plc_unit_type=='low' else 0)
        elif self.bitrate == '9k':
            if bit_per_frame == 90:   # 10ms  90bit
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 144 if config["use_complete_latent"] else 288
                        self.codebook_num = 36                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 144 if config["use_complete_latent"] else 288
                        self.codebook_num = 48                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 144 if config["use_complete_latent"] else 216
                        self.codebook_num = 18
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 144 if config["use_complete_latent"] else 216
                        self.codebook_num = 24                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 144 if config["use_complete_latent"] else 216
                        self.codebook_num = 9
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 144 if config["use_complete_latent"] else 216
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                    self.mask_chnls = 1 
            elif bit_per_frame == 45:  # 5ms  45bit  
                if config["combineVQ_frames"]==4:  # 120bit                    
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 72 if config["use_complete_latent"] else 288                            
                        self.codebook_num = 18
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 72 if config["use_complete_latent"] else 288 
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1                                             
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 72 if config["use_complete_latent"] else 216                            
                        self.codebook_num = 9
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 72 if config["use_complete_latent"] else 216
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                     
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1
                else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 90 if config["use_complete_latent"] else 120
                        self.codebook_num = 5
                        self.codebook_size = [512 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 90 if config["use_complete_latent"] else 120
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                    
                    if self.add_packet_loss:
                        self.mask_chnls = self.latent_dim if self.plc_unit_type=='full' else (1 if self.plc_unit_type=='low' else 0)
        elif self.bitrate == '12k':
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1
            if bit_per_frame == 120:   # 10ms  120bit
                if config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': #240bit
                        self.latent_dim = 168 if config["use_complete_latent"] else 312
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 320
                        self.codebook_num = 32                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete':  #120bit
                        self.latent_dim = 168 if config["use_complete_latent"] else 240
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 240
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                  
            elif bit_per_frame == 60:  # 5ms  60bit
                if config["combineVQ_frames"]==4:                       
                    if config["vq_rate"] == 'complete':  #240bit
                        self.latent_dim = 120 if config["use_complete_latent"] else 360                            
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 160 if config["use_complete_latent"] else 480 
                        self.codebook_num = 32
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                        
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete':   #120bit
                        self.latent_dim = 84 if config["use_complete_latent"] else 240                            
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 80 if config["use_complete_latent"] else 240
                        self.codebook_num = 16
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number         
                else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 80 if config["use_complete_latent"] else 120
                        self.codebook_num = 8
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 84 if config["use_complete_latent"] else 120
                        self.codebook_num = 12
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number   
            
    def init_codebook_size_24khz(self, config, bit_per_frame):
        if self.bitrate == '1k': 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_frame == 5:  # 5ms  5bit           
                if config["combineVQ_frames"]==8:                        
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 6                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                elif config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 2                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 3                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
            elif bit_per_frame == 10:  # 10ms  10bit                    
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 6                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 2
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 3                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number      
        elif self.bitrate == '3k': 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_frame == 15:  # 5ms  15bit
                # 2x(2^8x64) = 3.2 kbps                    
                if config["combineVQ_frames"]==8:                        
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 128 if config["use_complete_latent"] else 240
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                elif config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 6                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 3
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 2
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 3                        
                        self.codebook_size = [128 for i in range(self.codebook_num)]  # codeword number
            elif bit_per_frame == 30:  # 10ms  30bit                    
                if config["combineVQ_frames"]==4:  # 120bit
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2: # 60bit
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1   # 30bit
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 3
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                                       
        elif self.bitrate == '6k':
            if bit_per_frame == 60:   # 10ms  60bit
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 24                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 256 if config["use_complete_latent"] else 480
                        self.codebook_num = 32                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                    self.mask_chnls = 1 
            elif bit_per_frame == 30:  # 5ms  30bit  
                if config["combineVQ_frames"]==4:  # 120bit                    
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 120 if config["use_complete_latent"] else 240                            
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 128 if config["use_complete_latent"] else 240
                        self.codebook_num = 16
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1                                             
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 120 if config["use_complete_latent"] else 240                            
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 8
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                     
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1
                else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 3
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 4
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                    
                    if self.add_packet_loss:
                        self.mask_chnls = self.latent_dim if self.plc_unit_type=='full' else (1 if self.plc_unit_type=='low' else 0)
        elif self.bitrate == '9k':
            if bit_per_frame == 90:   # 10ms  90bit
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 252 if config["use_complete_latent"] else 468
                        self.codebook_num = 36                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 288 if config["use_complete_latent"] else 480
                        self.codebook_num = 48                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 252 if config["use_complete_latent"] else 468
                        self.codebook_num = 18
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 24                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 243 if config["use_complete_latent"] else 477
                        self.codebook_num = 9
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                    self.mask_chnls = 1 
            elif bit_per_frame == 45:  # 5ms  45bit  
                if config["combineVQ_frames"]==4:  # 180bit                    
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 126 if config["use_complete_latent"] else 234                            
                        self.codebook_num = 18
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1                                             
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete':   
                        self.latent_dim = 126 if config["use_complete_latent"] else 234                            
                        self.codebook_num = 9
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                     
                    if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                        self.mask_chnls = 1
                else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 5
                        self.codebook_size = [512 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                    
                    if self.add_packet_loss:
                        self.mask_chnls = self.latent_dim if self.plc_unit_type=='full' else (1 if self.plc_unit_type=='low' else 0)
        elif self.bitrate == '12k':
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1
            if bit_per_frame == 120:   # 10ms  120bit
                if config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': #240bit
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 256 if config["use_complete_latent"] else 480
                        self.codebook_num = 32                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete':  #120bit
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 480
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                  
            elif bit_per_frame == 60:  # 5ms  60bit
                if config["combineVQ_frames"]==4:                       
                    if config["vq_rate"] == 'complete':  #240bit
                        self.latent_dim = 120 if config["use_complete_latent"] else 240                            
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 128 if config["use_complete_latent"] else 256 
                        self.codebook_num = 32
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                        
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete':   #120bit
                        self.latent_dim = 120 if config["use_complete_latent"] else 240                            
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                
                    else: # 160 bit
                        self.latent_dim = 128 if config["use_complete_latent"] else 240
                        self.codebook_num = 16
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number         
                else:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 8
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 120 if config["use_complete_latent"] else 240
                        self.codebook_num = 12
                        self.codebook_size = [256 for i in range(self.codebook_num)]  # codeword number   
            
    def init_codebook_size_48khz(self, config, bit_per_frame, is_enhance_layer):
        bit_per_packet = bit_per_frame * config["combineVQ_frames"]
        if self.bitrate == '1k' and is_enhance_layer: 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_packet == 40:  # hop10ms/10bit/4frm or hop20ms/20bit/2frm  
                if config["vq_rate"] == 'complete': 
                    self.latent_dim = 160 if config["use_complete_latent"] else 288
                    self.codebook_num = 4                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 162 if config["use_complete_latent"] else 288
                    self.codebook_num = 6                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
            elif bit_per_packet == 20:  # hop10ms/10bit/2frm or hop20ms/20bit/1frm             
                if config["vq_rate"] == 'complete': 
                    self.latent_dim = 160 if config["use_complete_latent"] else 288
                    self.codebook_num = 2
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 162 if config["use_complete_latent"] else 288
                    self.codebook_num = 3                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
        elif self.bitrate == '2k' and is_enhance_layer: 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_packet == 80:  # hop10ms/20bit/4frm
                if config["vq_rate"] == 'complete': 
                    self.latent_dim = 160 if config["use_complete_latent"] else 288
                    self.codebook_num = 8                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 168 if config["use_complete_latent"] else 288
                    self.codebook_num = 12                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
            elif bit_per_packet == 40:  # hop10ms/20bit/2frm           
                if config["vq_rate"] == 'complete': 
                    self.latent_dim = 160 if config["use_complete_latent"] else 288
                    self.codebook_num = 4
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                else:
                    self.latent_dim = 162 if config["use_complete_latent"] else 288
                    self.codebook_num = 6                        
                    self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                      
        elif self.bitrate == '3k' and not is_enhance_layer: 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_frame == 30:  # 10ms  30bit                    
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 368 
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 3
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 4                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
        elif self.bitrate == '4k' and is_enhance_layer: 
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1            
            if bit_per_frame == 40:  # 10ms  40bit                    
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 160 if config["use_complete_latent"] else 288
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 160 if config["use_complete_latent"] else 300 
                        self.codebook_num = 20                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 160 if config["use_complete_latent"] else 288
                        self.codebook_num = 8
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 168 if config["use_complete_latent"] else 288
                        self.codebook_num = 12                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                         
        elif self.bitrate == '6k' and not is_enhance_layer:
            if bit_per_frame == 60:   # 10ms  60bit
                if config["combineVQ_frames"]==4:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 24                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 256 if config["use_complete_latent"] else 384
                        self.codebook_num = 32                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number  
                elif config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 368
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete': 
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 6
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 8                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                    self.mask_chnls = 1             
        elif self.bitrate == '12k' and not is_enhance_layer:
            if self.add_packet_loss and (not (self.plc_unit_type=='none')):
                self.mask_chnls = 1
            if bit_per_frame == 120:   # 10ms  120bit
                if config["combineVQ_frames"]==2:
                    if config["vq_rate"] == 'complete': #240bit
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 24
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 256 if config["use_complete_latent"] else 384
                        self.codebook_num = 32                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number 
                else: # 1
                    if config["vq_rate"] == 'complete':  #120bit
                        self.latent_dim = 240 if config["use_complete_latent"] else 360
                        self.codebook_num = 12
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number
                    else:
                        self.latent_dim = 240 if config["use_complete_latent"] else 368
                        self.codebook_num = 16                        
                        self.codebook_size = [1024 for i in range(self.codebook_num)]  # codeword number                                              
    
    def forward(self, inputs, loss_mask=None, epo=None): # (B, C, T, 1)
        return

    def vq_bottleneck(self, inputs, bitrate=None, loss_mask=None, epo=None): # (B, C, T, 1)
        result = {}
                   
        enc_feat = inputs   
        
        self.bitrate = self.config['bitrate'] if not is_bitrate_scalable(self.config) else bitrate

        
        bitrate = self.bitrate
        result["bitrate"] = bitrate
        # input shape: (B, L)           
                 
        batch_size, nb_frames = enc_feat.shape[0], enc_feat.shape[2]
        #enc_out_channels, enc_out_frm_size = enc_feat.shape[1], enc_feat.shape[3]
        #enc_feat = enc_feat.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(-1)  # (B,CF,T,1)
        enc_feat = self.conv1x1_1(enc_feat)  # [B,120,T,1]
        enc_feat_r = enc_feat.squeeze(-1).permute(0, 2, 1) #[B,T,C]
        
        if self.config["use_parallelvq"] and self.config["use_predictive"]:
            vq_out = self.quantize_parallel(enc_feat_r, epo=epo)
        else:
            vq_out = self.quantize(enc_feat_r, bitrate=bitrate, epo=epo)
        
        vq_feat = vq_out["vq_feat"].permute(0, 2, 1).unsqueeze(-1)  # [B,320,T,1]
        if self.add_packet_loss:
            assert(loss_mask is not None)
            size_feat = vq_feat.size()
            chnls = size_feat[1] if (self.plc_unit_type=='full' and self.combine_frames == 1) else 1
            loss_mask_in = loss_mask.to(torch.float32).repeat(1, chnls, 1, size_feat[-1]) # (B, C, T, 1) 
            vq_feat = vq_feat * (1.0 - loss_mask_in)
            if not (self.plc_unit_type=='none'):
                vq_feat_mask = torch.cat((vq_feat, loss_mask_in),1)                  
                vq_feat = self.conv1x1_2(vq_feat_mask) if not is_bitrate_scalable(self.config) else self.conv1x1_2[self.bitrate_list.index(bitrate)](vq_feat_mask)  # [B,320,T,1]

            else:                   
                vq_feat = self.conv1x1_2(vq_feat) if not is_bitrate_scalable(self.config) else self.conv1x1_2[self.bitrate_list.index(bitrate)](vq_feat)  # [B,320,T,1]

        else:                
            vq_feat = self.conv1x1_2(vq_feat) if not is_bitrate_scalable(self.config) else self.conv1x1_2[self.bitrate_list.index(bitrate)](vq_feat)  # [B,320,T,1]
        #vq_feat = vq_feat.squeeze(-1).reshape(batch_size, enc_out_channels, enc_out_frm_size, nb_frames).permute(0, 1, 3, 2)# [B,C,T,F]
        
        if 'quantization_inds' in vq_out.keys():
            result.update({"quantization_inds": vq_out["quantization_inds"]})
        if 'commitment_loss' in vq_out.keys():
            result.update({"commitment_loss": vq_out["commitment_loss"]})
        if 'codebook_usage' in vq_out.keys():
            result.update({"codebook_usage": vq_out["codebook_usage"]})
        if "predictive_loss" in vq_out.keys():
            result.update({"predictive_loss": vq_out["predictive_loss"]})
        if "feature_sparse_loss" in vq_out.keys():
            result.update({"feature_sparse_loss": vq_out["feature_sparse_loss"]})
        if self.config["vq_type"] == 'Gumbel':
            if 'prob_perplexity' in vq_out.keys():
                result.update({"prob_perplexity_list": vq_out["prob_perplexity_list"]})
            if 'code_perplexity' in vq_out.keys():
                result.update({"code_perplexity_list": vq_out["code_perplexity_list"]})
            if 'entropy_loss' in vq_out.keys():
                result.update({"entropy_loss": vq_out["entropy_loss"]})
            if 'entropy' in vq_out.keys():
                result.update({"entropy": vq_out["entropy"]})  # entropy for current batch
                result.update({"entropy_avg": vq_out["entropy_avg"]})  # entropy average from 1st batch to current batch
        else:
            result.update({"perplexity_list": vq_out["prob_perplexity_list"]})        

        result["vq_feat"] = vq_feat        

        return result  

    def quantize(self, enc_feat, bitrate=None, epo=None):
        ### vector quantization
        #input shape [B,T,C]
        result = {}
        vq_layer_out = []
        vq_inds_out = []
        prob_perplexity_list = []
        code_perplexity_list = []
        entropy_list = []
        entropy_avg_list = []
        commitment_loss = 0
        codebook_usage = 0
        predictive_loss = 0
        sparse_loss = 0
        entropy_loss = 0
        #self.vq_in_list = []
        if self.config["use_entropy_loss"]:
            target_entropy_per_vqlayer = torch.tensor(self.target_entropy/self.codebook_num).to(enc_feat)
            fuzz_entropy_per_vqlayer = torch.tensor(self.entropy_fuzz/self.codebook_num).to(enc_feat)       
        
        B, T, channels = enc_feat.shape # B,T,C'
        combine_frames = self.combine_frames           
        enc_feat_combine = enc_feat.reshape(B, T// combine_frames, combine_frames, channels) # B,T/2,2,C'
            
        if 0:
            # calculate temporal correlation coefficient
            mean = torch.mean(enc_feat_combine[:,:-1,:,:].reshape(1,-1,combine_frames*channels), dim=1)
            cross_correlation = torch.sum(torch.matmul(enc_feat_combine[:,:-1,:,:].reshape(-1, 1, combine_frames*channels)-mean, enc_feat_combine[:,1:,:,:].reshape(-1, combine_frames*channels, 1)-mean.reshape(-1,1)),dim=0)
            self_correlation = torch.sum(torch.matmul(enc_feat_combine[:,:-1,:,:].reshape(-1, 1, combine_frames*channels)-mean, enc_feat_combine[:,:-1,:,:].reshape(-1, combine_frames*channels, 1)-mean.reshape(-1,1)),dim=0)
            corr_coef = cross_correlation/self_correlation
            print('coeff: {}, cross: {}, self: {}'.format(corr_coef, cross_correlation, self_correlation))
            print(mean[0:5])            
    
        #if config["use_scale_float16"]: #todo
        #    scale = enc_feat_combine.reshape(B, T// combine_frames, -1)            

        if self.config["use_parallelvq"]:
            enc_feat_combine = enc_feat_combine.permute(0,1,3,2).reshape(B, T//combine_frames,-1) # B,T/2,C'*2
            enc_feat_combine = torch.split(enc_feat_combine, channels*combine_frames//self.num_its_para, dim=-1)  
            for layer_i in range(len(self.vq_layer_list)):
                vq_layer = self.vq_layer_list[layer_i]
                vq_in = enc_feat_combine[layer_i].reshape(B, T // combine_frames, -1)                
                if self.config["use_entropy_loss"]:
                    if self.config["use_predictive"]:
                        vq_out = vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer, epo=epo)
                    else:
                        vq_out = vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer)
                else:
                    if self.config["use_predictive"]:
                        vq_out = vq_layer(vq_in, epo=epo)
                    else:
                        vq_out = vq_layer(vq_in)
                vq_layer_out.append(vq_out["quantized_feature"])  # [B,T/2,channel*2//num_its_para]
                vq_inds_out.append(vq_out["quantization_inds"]) # [B,T/2,codebook_num//num_its_para]
                if 'entropy' in vq_out.keys(): # softmax prob
                    entropy_list.extend(torch.split(vq_out["entropy"], 1, dim=-1)) # entropy for current batch
                    entropy_avg_list.append(vq_out["entropy_avg"]) # entropy for current epoch
                if 'prob_perplexity' in vq_out.keys(): # soft prob
                    prob_perplexity_list.extend(torch.split(vq_out["prob_perplexity"], 1, dim=-1))
                if 'code_perplexity' in vq_out.keys(): # hard prob
                    code_perplexity_list.extend(torch.split(vq_out["code_perplexity"], 1, dim=-1))
                if 'codebook_usage' in vq_out.keys():
                    codebook_usage += vq_out["codebook_usage"]
                if 'commitment_loss' in vq_out.keys():
                    commitment_loss += vq_out["commitment_loss"] # commitment loss
                if 'entropy_loss' in vq_out.keys():
                    entropy_loss += vq_out["entropy_loss"]                
                if "predictive_loss" in vq_out.keys():
                    predictive_loss += vq_out["predictive_loss"]
                if "feature_sparse_loss" in vq_out.keys():
                    sparse_loss += vq_out["feature_sparse_loss"]
                    
            if self.combine_frames > 1:
                vq_feat = torch.cat(vq_layer_out, dim=-1).reshape(B, T//combine_frames, channels, combine_frames).permute(0,1,3,2) # B,T/2,2,C'                        
                result["vq_feat"] = vq_feat.reshape(B,T,channels)
            else:
                result["vq_feat"] =  torch.cat(vq_layer_out, dim=-1)  # B,T,320
        else:
            enc_feat_combine = torch.split(enc_feat_combine, channels//self.codebook_num,dim=-1)  # codebook_num,B,T/2,2,C'//codebook_num
            valid_vq_list = self.vq_layer_list if not is_bitrate_scalable(self.config) else self.vq_layer_list[:self.valid_groups_dict[bitrate]]
            for layer_i in range(len(valid_vq_list)):
                vq_layer = self.vq_layer_list[layer_i]
                if self.combine_frames > 1:
                    vq_in = enc_feat_combine[layer_i].reshape(B, T // combine_frames, -1)
                else:
                    vq_in = enc_feat[:, : ,layer_i*self.codebook_dim:(layer_i+1)*self.codebook_dim]
                #self.vq_in_list.append(vq_in.reshape(B*T // combine_frames,-1)) # just for visual
                if self.config["use_entropy_loss"]:
                    if self.config["use_predictive"]:
                        vq_out = vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer, epo=epo)
                    else:
                        vq_out = vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer)
                else:
                    if self.config["use_predictive"]:
                        vq_out = vq_layer(vq_in, epo=epo)
                    else:
                        vq_out = vq_layer(vq_in)
                if self.combine_frames > 1:
                    vq_layer_out.append(vq_out["quantized_feature"].reshape(B,T//combine_frames,combine_frames,channels//self.codebook_num))  # [B,T/2,2,C'//codebook_num]
                else:
                    vq_layer_out.append(vq_out["quantized_feature"])  # [B,T,C'//codebook_num]
                vq_inds_out.append(vq_out["quantization_inds"]) # [B,T,1]
                if 'entropy' in vq_out.keys(): # softmax prob
                    entropy_list.append(vq_out["entropy"]) # entropy for current batch
                    entropy_avg_list.append(vq_out["entropy_avg"]) # entropy for current epoch
                if 'prob_perplexity' in vq_out.keys(): # soft prob
                    prob_perplexity_list.append(vq_out["prob_perplexity"])
                if 'code_perplexity' in vq_out.keys(): # hard prob
                    code_perplexity_list.append(vq_out["code_perplexity"])
                if 'codebook_usage' in vq_out.keys():
                    codebook_usage += vq_out["codebook_usage"]
                if 'commitment_loss' in vq_out.keys():
                    commitment_loss += vq_out["commitment_loss"] # commitment loss
                if 'entropy_loss' in vq_out.keys():
                    entropy_loss += vq_out["entropy_loss"]                
                if "predictive_loss" in vq_out.keys():
                    predictive_loss += vq_out["predictive_loss"]
                if "feature_sparse_loss" in vq_out.keys():
                    sparse_loss += vq_out["feature_sparse_loss"]
                    
            if self.combine_frames > 1:
                vq_feat = torch.cat(vq_layer_out, dim=-1) # B,T/2,2,C'
                if is_bitrate_scalable(self.config):
                    zero_groups_num = len(self.vq_layer_list) - self.valid_groups_dict[bitrate]
                    vq_feat = F.pad(vq_feat, (0, zero_groups_num * channels // self.codebook_num))
                        
                result["vq_feat"] = vq_feat.reshape(B,T,channels)
            else:
                result["vq_feat"] =  torch.cat(vq_layer_out, dim=-1)  # B,T,320

        codebook_usage /= len(self.vq_layer_list)
        commitment_loss /= len(self.vq_layer_list)
        predictive_loss /= len(self.vq_layer_list)
        sparse_loss /= len(self.vq_layer_list)
        
        result["quantization_inds"] =  torch.cat(vq_inds_out,dim=-1)  # B,T,codebook_num
        if 'prob_perplexity' in vq_out.keys(): # soft prob
            result["prob_perplexity_list"] = prob_perplexity_list  # [codebook_num]
        if 'code_perplexity' in vq_out.keys(): # hard prob
            result["code_perplexity_list"] = code_perplexity_list  # [codebook_num]
        if 'codebook_usage' in vq_out.keys():
            result["codebook_usage"] = codebook_usage  # [1,]  avg_codebook_usage
        if 'commitment_loss' in vq_out.keys():
            result["commitment_loss"] = commitment_loss # [1,]  avg_vqloss
        if 'entropy_loss' in vq_out.keys():
            result["entropy_loss"] = entropy_loss #
        if 'entropy' in vq_out.keys():
            result["entropy"] = entropy_list  #
            result["entropy_avg"] = entropy_avg_list  #
        if "predictive_loss" in vq_out.keys():
            result["predictive_loss"] = predictive_loss
        if "feature_sparse_loss" in vq_out.keys():
            result["feature_sparse_loss"] = sparse_loss            

        return result
        
    def quantize_parallel(self, enc_feat, epo=None):
        ### vector quantization
        #input shape [B,T,C]
        result = {}
        vq_layer_out = []
        vq_inds_out = []
        prob_perplexity_list = []
        code_perplexity_list = []
        entropy_list = []
        entropy_avg_list = []
        commitment_loss = 0
        codebook_usage = 0
        predictive_loss = 0
        entropy_loss = 0
        #self.vq_in_list = []
        if self.config["use_entropy_loss"]:
            target_entropy_per_vqlayer = torch.tensor(self.target_entropy/self.codebook_num).to(enc_feat)
            fuzz_entropy_per_vqlayer = torch.tensor(self.entropy_fuzz/self.codebook_num).to(enc_feat)       
            
        B, T, channels = enc_feat.shape # B,T,C'
        combine_frames = self.combine_frames
        enc_feat_combine = enc_feat.reshape(B, T//combine_frames, combine_frames, channels).permute(0,1,3,2).reshape(B, T//combine_frames,-1) # B,T/2,C'*2

        vq_in = enc_feat_combine
        if self.config["use_entropy_loss"]:
            if self.config["use_predictive"]:
                if self.training and self.config["prediction_stage"] == '1':
                    vq_out = self.vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer, epo=epo, ar_signal=vq_in)
                elif self.training and self.config["prediction_stage"] == '3':
                    vq_temp = self.vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer, epo=epo, ar_signal=None)
                    ar_signal = vq_temp["quantized_feature"].detach()
                    vq_out = self.vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer, epo=epo, ar_signal=ar_signal)
                else:
                    vq_out = self.vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer, epo=epo)
            else:
                vq_out = self.vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer)
        else:
            if self.config["use_predictive"]:
                vq_out = self.vq_layer(vq_in, epo=epo)
            else:
                vq_out = self.vq_layer(vq_in)
        result["vq_feat"] = vq_out["quantized_feature"].reshape(B, T//combine_frames, channels, combine_frames).permute(0,1,3,2).reshape(B, T, channels)       
        result["quantization_inds"] = vq_out["quantization_inds"]  # B,T,codebook_num
        if 'prob_perplexity' in vq_out.keys(): # soft prob
            result["prob_perplexity_list"] = vq_out["prob_perplexity"]  # [codebook_num]
        if 'code_perplexity' in vq_out.keys(): # hard prob
            result["code_perplexity_list"] = vq_out["code_perplexity"]  # [codebook_num]
        if 'commitment_loss' in vq_out.keys():
            result["commitment_loss"] = vq_out["commitment_loss"] # [1,]  avg_vqloss
        if 'entropy_loss' in vq_out.keys():
            result["entropy_loss"] = vq_out["entropy_loss"] #
        if 'entropy' in vq_out.keys():
            result["entropy"] = vq_out["entropy"]  #entropy for current batch
            result["entropy_avg"] = vq_out["entropy_avg"]  #entropy for current epoch
        if "predictive_loss" in vq_out.keys():
            result["predictive_loss"] = vq_out["predictive_loss"]
        if "feature_sparse_loss" in vq_out.keys():
            result["feature_sparse_loss"] = vq_out["feature_sparse_loss"]

        return result

    def dequantize(self, vq_inds):
        vq_layer_out = []
        if self.combine_frames > 1:
            B,T,_ = vq_inds.shape # B,T,C'
            combine_frames = self.combine_frames
        if self.config["use_parallelvq"]:
            vq_inds_list = torch.split(vq_inds, self.codebook_num//self.num_its_para, dim=-1)  
            for layer_i in range(len(self.vq_layer_list)):
                vq_layer = self.vq_layer_list[layer_i]
                vq_out = vq_layer.dequantize(vq_inds_list[:,:,layer_i])
                vq_layer_out.append(vq_out)  
            vq_feat = torch.cat(vq_layer_out, dim=-1)  
            if self.combine_frames > 1:
                vq_feat = vq_feat.reshape(B,T,-1,combine_frames).permute(0,1,3,2).reshape(B,T*combine_frames,-1)
        else:
            for layer_i in range(len(self.vq_layer_list)):
                vq_layer = self.vq_layer_list[layer_i]
                vq_out = vq_layer.dequantize(vq_inds[:,:,layer_i])
                if self.combine_frames > 1:
                    vq_layer_out.append(vq_out.reshape(B, T, combine_frames,-1))  # [B,T/2,2,C'//codebook_num]
                else:
                    vq_layer_out.append(vq_out)  # [B,T,C'//codebook_num]
            vq_feat = torch.cat(vq_layer_out, dim=-1)  # B,T,C'
            if self.combine_frames > 1:
                vq_feat = vq_feat.reshape(B,T*combine_frames,-1)

        return vq_feat

    def dequantize_from_logits(self, logits):
        vq_layer_out = []
        if self.combine_frames > 1:
            B,T,_,_ = logits.shape # B,T,c,q
            combine_frames = self.combine_frames
        if self.config["use_parallelvq"]:
            vq_logits_list = torch.split(logits, self.codebook_num//self.num_its_para, dim=-1)  
            for layer_i in range(len(self.vq_layer_list)):
                vq_layer = self.vq_layer_list[layer_i]
                vq_out = vq_layer.dequantize_from_logits(vq_logits_list[:,:,:,layer_i])
                vq_layer_out.append(vq_out)  
            vq_feat = torch.cat(vq_layer_out, dim=-1)  
            if self.combine_frames > 1:
                vq_feat = vq_feat.reshape(B,T,-1,combine_frames).permute(0,1,3,2).reshape(B,T*combine_frames,-1)
        else:
            for layer_i in range(len(self.vq_layer_list)):
                vq_layer = self.vq_layer_list[layer_i]
                vq_out = vq_layer.dequantize_from_logits(logits[:,:,:,layer_i])
                if self.combine_frames > 1:
                    vq_layer_out.append(vq_out.reshape(B, T, combine_frames,-1))  # [B,T/2,2,C'//codebook_num]
                else:
                    vq_layer_out.append(vq_out)  # [B,T,C'//codebook_num]
            vq_feat = torch.cat(vq_layer_out, dim=-1)  # B,T,C'
            if self.combine_frames > 1:
                vq_feat = vq_feat.reshape(B,T*combine_frames,-1)

        return vq_feat
    
    def decode_vq_bottleneck(self, vq_inds):
        batch_size, nb_frames = vq_inds.shape[0], vq_inds.shape[1]
        vq_feat = self.dequantize(vq_inds)      # [B, T, C]  
        vq_feat = vq_feat.permute(0, 2, 1).unsqueeze(-1)  # [B,320,T,1]                       
        vq_feat = self.conv1x1_2(vq_feat)  # [B,320,T,1]
        #vq_feat = vq_feat.squeeze(-1).reshape(batch_size, enc_out_channels, enc_out_frm_size, nb_frames).permute(0, 1, 3, 2)# [B,C,T,F]       
        return vq_feat
        
    def set_network_entropy_target(self, bitrate, fuzz, hop_dur):        
        bitrate_per_vq_layer = bitrate / self.codebook_num
        self.target_entropy = 0
        self.entropy_fuzz = 0
        fuzz_per_vq_layer = fuzz / self.codebook_num
        for ii in range(self.codebook_num):
            self.target_entropy += bitrate_to_entropy_2(bitrate_per_vq_layer, hop_dur)
            self.entropy_fuzz += bitrate_to_entropy_2(fuzz_per_vq_layer, hop_dur)

    def reset_entropy_hists_train(self):
        if self.config["use_parallelvq"] and self.config["use_predictive"]:
            self.vq_layer.entropy_avg_train.reset()
        else:
            for vq_layer in self.vq_layer_list:
                vq_layer.entropy_avg_train.reset()

    def reset_entropy_hists_eval(self):
        if self.config["use_parallelvq"] and self.config["use_predictive"]:
            self.vq_layer.entropy_avg_eval.reset()
        else:
            for vq_layer in self.vq_layer_list:
                vq_layer.entropy_avg_eval.reset()

    def get_overall_entropy_avg_train(self, bitrate=' '):
        if self.config["use_parallelvq"] and self.config["use_predictive"]:
            return [self.vq_layer.entropy_avg_train.avg]            
        avgs = []
        valid_vq_list = self.vq_layer_list if not is_bitrate_scalable(self.config) else self.vq_layer_list[:self.valid_groups_dict[bitrate]]
        for vq_layer in valid_vq_list:
            avgs.append(vq_layer.entropy_avg_train.avg)
        return [torch.stack(avgs, dim=0).sum(dim=0)]

    def get_overall_entropy_avg_eval(self):
        if self.config["use_parallelvq"] and self.config["use_predictive"]:
            return [self.vq_layer.entropy_avg_eval.avg]
        avgs = []
        for vq_layer in self.vq_layer_list:
            avgs.append(vq_layer.entropy_avg_eval.avg)
        return [torch.stack(avgs, dim=0).sum(dim=0)]
        
    def update_temperature_gumbel(self, cur_iter):    
        if 'Gumbel' in self.config["vq_type"]:
            if self.config["use_parallelvq"] and self.config["use_predictive"]:
                self.vq_layer.temp_updates(cur_iter)
            else:
                for vq_layer in self.vq_layer_list:
                    vq_layer.temp_updates(cur_iter)
        

    

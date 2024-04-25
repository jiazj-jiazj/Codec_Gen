import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# import matplotlib.pyplot as plt
import torch.distributed as dist


class VQEmbeddingEMA_Predictive_DDP(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, decay=0.999, epsilon=1e-5, config=None):
        super(VQEmbeddingEMA_Predictive_DDP, self).__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.init = False
        self.ema_weight = None
        self.ema_count = None
        self.threshold = 1.0
        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer('embedding', embedding)
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.config = config
        
        self.context_len = 7
        self.context_gen = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=self.context_len, stride=1, bias=False)
        
        if config["fuse_type"] == 'conv':
             self.fuse_conv1 = nn.Conv1d(2*embedding_dim, embedding_dim, kernel_size=1, stride=1, bias=False)
             self.nonlinearity1 = nn.PReLU(init=0.5) 
             self.fuse_conv2 = nn.Conv1d(2*embedding_dim, embedding_dim, kernel_size=1, stride=1, bias=False)
             self.nonlinearity2 = nn.PReLU(init=0.5) 
             
        # if self.config is not None and self.config["use_attended_vq"]:
            # self.query = nn.Conv1d(embedding_dim, embedding_dim, 1, 1, bias=True)
            # self.key = nn.Conv1d(embedding_dim, embedding_dim, 1, 1, bias=True)
            # self.nonlinear = nn.relu()

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_embeddings: #todo
            n_repeats = (self.n_embeddings + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_embedding(self, x_flat):  
        decay, embedding_dim, n_embeddings = self.decay, self.embedding_dim, self.n_embeddings
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x_flat)
        embedding_rand = y[torch.randperm(y.shape[0])][:n_embeddings]
        if torch.cuda.is_available():
            dist.broadcast(embedding_rand, 0)
        self.embedding = embedding_rand
        assert self.embedding.shape == (n_embeddings, embedding_dim)
        self.ema_weight = self.embedding
        self.ema_count = torch.ones(n_embeddings, device=self.embedding.device)

    def update_embedding(self, x_flat, indices):
        decay, embedding_dim, n_embeddings = self.decay, self.embedding_dim, self.n_embeddings
        with torch.no_grad():
            # Calculate new centres
            encodings = F.one_hot(indices, n_embeddings).float()
            embedding_sum = torch.matmul(encodings.t(), x_flat)  # 256,64 single GPU batch
            embedding_count = torch.sum(encodings, dim=0)  # 256
            y = self._tile(x_flat)
            embedding_rand = y[torch.randperm(y.shape[0])][:n_embeddings]
            # print("embedding_rand{} ".format(embedding_rand))
            if torch.cuda.is_available():
                dist.barrier()
                # print("codebook_sync!!!!") # print("embedding_rand {} on device {} ".format(embedding_sum[0][0], embedding_sum[0][0].get_device()))
                dist.broadcast(embedding_rand, 0)
                dist.all_reduce(embedding_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(embedding_count, op=dist.ReduceOp.SUM)
            # Update centres
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * embedding_count  # (256,)
            # n = torch.sum(self.ema_count)
            # self.ema_count = (self.ema_count + self.epsilon) / (n + n_embeddings * self.epsilon) * n
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * embedding_sum
            usage = (self.ema_count.view(n_embeddings, 1) >= self.threshold).float()
            self.embedding = usage * (self.ema_weight.view(n_embeddings, embedding_dim) / self.ema_count.view(n_embeddings, 1)) + (1 - usage) * embedding_rand

    # def quantize(self, x):
        # M, D = self.embedding.size()
        # x_flat = x.detach().reshape(-1, D)

        # distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                # torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                # x_flat, self.embedding.t(),
                                # alpha=-2.0, beta=1.0)

        # indices = torch.argmin(distances.float(), dim=-1)
        # quantized = F.embedding(indices, self.embedding)
        # quantized = quantized.view_as(x)
        # return quantized, indices

    # def dequantize(self,indices):
        # ### 1
        # bsz,tsz= indices.shape
        # quantized = F.embedding(indices.reshape(bsz*tsz,-1), self.embedding)
        # quantized = quantized.reshape(bsz,tsz,-1)
        # return quantized

    def forward(self, x, epo=None):   #[B, T, C]
        output={}
        M, D = self.embedding.size()  #512 ,64
        x_flat = x.detach()  #[B, T, C]
        bsz,tsz,csz = x.shape       
        
        if self.training and (self.config["prediction_stage"] == '1'):
            ctx_in = F.pad(x[:, :-1, :].permute(0, 2, 1), (1+self.context_len-1, 0)).detach()
            uni_noise = torch.randn(ctx_in.size()).to(ctx_in)
            scale = torch.sqrt(torch.mean(torch.square(ctx_in), dim=(1,2), keepdim=True))
            ctx_in_noisy = ctx_in + 0.5*scale*uni_noise*(0.5**(epo/10.0)) # reduced per 10 epochs, scale-dependent           
            ctx_out = self.context_gen(ctx_in_noisy)
            #ctx_out = self.context_gen(ctx_in)
            #ctx_out = ctx_in_noisy  # only use one previous feature
            if self.config["fuse_type"] == 'conv':
                merge_out = self.nonlinearity1(self.fuse_conv1(torch.cat((x, ctx_out), dim=1)))
                cur_x = merge_out.permute(0, 2, 1)
            elif self.config["fuse_type"] == 'res':
                context = ctx_out.permute(0, 2, 1)            
                cur_x = x - context   #[B, T, C]
            cur_x_detach = cur_x.detach().reshape(-1, D)
            if self.training and not self.init: 
                if self.config["vq_initial"]:
                    print('initialize codebook using features----------------------------')
                    self.init_embedding(cur_x_detach)
                else:            
                    self.init = True
                    self.ema_weight = self.embedding ## load vq embedding
                    self.ema_count = torch.ones(self.n_embeddings, device=self.embedding.device)            

            distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1)+torch.sum(cur_x_detach ** 2, dim=-1, keepdim=True), cur_x_detach, self.embedding.t(),alpha=-2.0, beta=1.0) # [BT, M]
            indices = torch.argmin(distances.float(), dim=-1)  #(2002,)
            quantized = F.embedding(indices, self.embedding)
            quantized = quantized.view_as(cur_x)            

            if self.training:
                self.update_embedding(cur_x_detach, indices)
                
            #Passthrough
            if self.config["fuse_type"] == 'conv':
                temp = cur_x + (quantized - cur_x).detach() 
                merge_out = self.nonlinearity2(self.fuse_conv2(torch.cat((temp.permute(0, 2, 1), ctx_out), dim=1)))
                q_output = merge_out.permute(0, 2, 1) # [B, T, D]
            elif self.config["fuse_type"] == 'res':
                q_output = x + (quantized - cur_x).detach() # cur_x + (quantized - cur_x).detach() + context[:,ii,:]
            indices_matrix = indices
            quantized_matrix = quantized            
        else:
            context = torch.zeros(x.shape).to(x)
            q_output = torch.zeros(x.shape).to(x)
            indices_list = []
            quantized_list = []
            for ii in range(tsz):
                if self.config["fuse_type"] == 'conv':
                    merge_out = self.nonlinearity1(self.fuse_conv1(torch.cat((x[:,ii:ii+1,:].permute(0, 2, 1), context[:,ii:ii+1,:].permute(0, 2, 1)), dim=1)))
                    cur_x = merge_out.squeeze(dim=-1)                   
                elif self.config["fuse_type"] == 'res':
                    cur_x = x[:,ii,:] - context[:,ii,:]   #[B, T, C]
                cur_x_detach = cur_x.detach()
                if self.training and not self.init:                
                    self.init = True
                    self.ema_weight = self.embedding ## load vq embedding
                    self.ema_count = torch.ones(self.n_embeddings, device=self.embedding.device)            

                distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1)+torch.sum(cur_x_detach ** 2, dim=-1, keepdim=True), cur_x_detach, self.embedding.t(),alpha=-2.0, beta=1.0) # [BT, M]
                indices = torch.argmin(distances.float(), dim=-1)  #(2002,)
                quantized = F.embedding(indices, self.embedding)
                quantized = quantized.view_as(cur_x)
                indices_list.append(indices)
                quantized_list.append(quantized)

                if self.training:
                    self.update_embedding(cur_x_detach, indices)
                    
                #Passthrough
                if self.config["fuse_type"] == 'conv': 
                    temp = cur_x + (quantized - cur_x).detach()           
                    merge_out = self.nonlinearity2(self.fuse_conv2(torch.cat((temp.unsqueeze(dim=-1), context[:,ii:ii+1,:].permute(0, 2, 1)), dim=1)))
                    q_output[:,ii:ii+1,:] = merge_out.permute(0, 2, 1)
                elif self.config["fuse_type"] == 'res':
                    q_output[:,ii,:] = x[:,ii,:] + (quantized - cur_x).detach() # cur_x + (quantized - cur_x).detach() + context[:,ii,:] 
                if ii+1 < tsz:
                    if ii < self.context_len:
                        ctx_in = F.pad(q_output[:,0:ii,:].permute(0, 2, 1), (self.context_len-ii, 0))
                    else:
                        ctx_in = q_output[:,ii-self.context_len:ii,:].permute(0, 2, 1)
                    ctx_out = self.context_gen(ctx_in.detach())  #self.context_gen(ctx_in)
                    #ctx_out = ctx_in.detach()  # only use one previous feature
                    context[:,ii+1:ii+2,:] = ctx_out.permute(0, 2, 1)
                
            indices_matrix = torch.stack(indices_list, dim=1)
            quantized_matrix = torch.stack(quantized_list, dim=1)
            
        commitment_cost = F.mse_loss(x - context.detach(), quantized_matrix.detach())        
            
        encodings = F.one_hot(indices_matrix, self.n_embeddings).float()
        embedding_count = torch.sum(encodings, dim=0)  # 256
        used_curr = (embedding_count >= self.threshold).sum().float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))       
               
        output["quantized_feature"] = q_output
        output["quantization_inds"] = indices_matrix.reshape(bsz,tsz).unsqueeze(-1) #todo
        output["prob_perplexity"] = used_curr
        output["codebook_usage"] = perplexity
        output["commitment_loss"] = commitment_cost
        if self.config["use_predictive_loss"]:
            output["predictive_loss"] = F.mse_loss(context, x.detach())
        else:
            output["predictive_loss"] = F.mse_loss(context.detach(), x.detach()) 

        return output


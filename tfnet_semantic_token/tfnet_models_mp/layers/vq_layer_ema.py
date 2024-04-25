import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# import matplotlib.pyplot as plt
import torch.distributed as dist


class VQEmbeddingEMA_DDP(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5, config=None):
        super(VQEmbeddingEMA_DDP, self).__init__()
        self.commitment_cost = commitment_cost
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

    def dequantize(self,indices):
        ### 1
        bsz,tsz= indices.shape
        quantized = F.embedding(indices.reshape(bsz*tsz,-1), self.embedding)
        quantized = quantized.reshape(bsz,tsz,-1)
        return quantized

    def forward(self, x):
        output={}
        M, D = self.embedding.size()  #512 ,64
        x_flat = x.detach().reshape(-1, D)  #[B, T, C]
        bsz,tsz,_ = x.shape
        if self.training and not self.init:
            if self.config["vq_initial"]:
                print('initialize codebook using features----------------------------')
                self.init_embedding(x_flat)
            else:
                self.init = True
                self.ema_weight = self.embedding ## load vq embedding
                self.ema_count = torch.ones(self.n_embeddings, device=self.embedding.device)
        # Calculate latent code x_l

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +torch.sum(x_flat ** 2, dim=1, keepdim=True),x_flat, self.embedding.t(),alpha=-2.0, beta=1.0)
        indices = torch.argmin(distances.float(), dim=-1)  #(2002,)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.dict = self.update_embedding(x_flat, indices)

        encodings = F.one_hot(indices, self.n_embeddings).float()
        embedding_count = torch.sum(encodings, dim=0)  # 256
        used_curr = (embedding_count >= self.threshold).sum().float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        commitment_cost = F.mse_loss(x, quantized.detach())
        # loss = self.commitment_cost * e_latent_loss
        #Passthrough
        quantized = x + (quantized - x).detach()
        output["quantized_feature"] = quantized
        output["quantization_inds"] = indices.reshape(bsz,tsz).unsqueeze(-1) #todo
        output["prob_perplexity"] = used_curr
        output["codebook_usage"] = perplexity
        output["commitment_loss"] = commitment_cost

        return output


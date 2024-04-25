import torch
from torch import nn
from torch.nn import functional as F
import random
class KMeans(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, epsilon=1e-5):
        super(KMeans, self).__init__()
        self.epsilon = epsilon

        init_bound = 1 / 256
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.cluster_num = n_embeddings
        self.cluster_dim = embedding_dim

    def codebook_lookup(self,x_flat):
        ### 1
        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) + torch.sum(x_flat ** 2, dim=1, keepdim=True),x_flat, self.embedding.t(), alpha=-2.0, beta=1.0)
        self.indices = torch.argmin(distances.float(), dim=-1)
        self.encodings = F.one_hot(self.indices, self.cluster_num).float()

    def codebook_init(self,x_flat):
        ### 1
        # random initial clustering
        random_indices = torch.tensor(random.sample(range(x_flat.shape[0]), self.cluster_num))
        self.embedding = x_flat[random_indices]

    def forward(self, iter, x_flat):
        if iter == 0 :
            self.codebook_init(x_flat)
        self.codebook_lookup(x_flat)
        cluster_distortion = self.codebook_distortion(x_flat)
        self.codebook_update(x_flat,self.encodings)
        return cluster_distortion

    def codebook_update(self,x_flat,encodings):
        self.ema_count = torch.sum(encodings, dim=0)  #(256
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + self.cluster_num * self.epsilon) * n
        self.ema_weight  = torch.matmul(encodings.t(), x_flat)
        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

    def codebook_distortion(self,x_flat):
        quantized = F.embedding(self.indices, self.embedding)
        cluster_distortion = F.mse_loss(x_flat, quantized).numpy()
        return cluster_distortion


class Moving_KMeans(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, epsilon=1e-5,decay=0.9):
        super(KMeans, self).__init__()
        self.epsilon = epsilon
        self.decay = decay

        init_bound = 1 / 256
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.cluster_num = n_embeddings
        self.cluster_dim = embedding_dim

    def codebook_lookup(self,x_flat):
        ### 1
        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) + torch.sum(x_flat ** 2, dim=1, keepdim=True),x_flat, self.embedding.t(), alpha=-2.0, beta=1.0)
        self.indices = torch.argmin(distances.float(), dim=-1)
        self.encodings = F.one_hot(self.indices, self.cluster_num).float()

    def codebook_init(self,x_flat):
        ### 1
        # random initial clustering
        random_indices = torch.tensor(random.sample(range(x_flat.shape[0]), self.cluster_num))
        self.embedding = x_flat[random_indices]

    def forward(self, iter, x_flat):
        if iter == 0 :
            self.codebook_init(x_flat)
        self.codebook_lookup(x_flat)
        cluster_distortion = self.codebook_distortion(x_flat)
        self.codebook_update(x_flat,self.encodings)
        return cluster_distortion

    def codebook_update(self,x_flat,encodings):
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
        dw = torch.matmul(encodings.t(), x_flat)
        self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

    def codebook_distortion(self,x_flat):
        quantized = F.embedding(self.indices, self.embedding)
        cluster_distortion = F.mse_loss(x_flat, quantized).numpy()
        return cluster_distortion
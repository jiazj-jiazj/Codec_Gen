import torch
import torch.nn as nn
import torch.nn.functional as F
from tfnet_semantic_token.utils.tools import AverageMeter
from .gumbel_max_pytorch import gumbel_softmax

class GumbelVectorQuantizer(nn.Module):
    def __init__(self, config, input_dim, n_embeddings, groups, combine_groups, weight_proj_channel=-1):
        """Vector quantization using gumbel softmax
        Args:
            input_dim: input dimension (channels)
            n_embeddings: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
        """
        super().__init__()

        self.groups = config["groups"]
        self.combine_groups = config["combine_groups"]
        self.input_dim = input_dim  # 120
        self.n_embeddings = n_embeddings  # 128
        vq_dim = input_dim
        assert (
                vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        embedding_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.embedding = nn.Parameter(torch.FloatTensor(1, num_groups * n_embeddings, embedding_dim))  # 1,1024,64
        nn.init.uniform_(self.embedding, a=-1.0, b=1.0)

        self.max_temp, self.min_temp, self.temp_decay = config["temperature"]
        self.curr_temp = self.max_temp
        self.codebook_indices = None
        self.config = config

        self.entropy_avg_train = AverageMeter()
        self.entropy_avg_eval = AverageMeter()
        self.code_entropy = 0
        self.tau = 0
        self.tau2 = 0.5
        self.alpha = config["dist_to_logits_alpha"]
        self._eps = torch.tensor(1e-7)
        if config["use_learnable_feat_cprs"]:
            self.feat_power_cprs = nn.Parameter(torch.FloatTensor(1))
            nn.init.constant_(self.feat_power_cprs, 0.5)

    def temp_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates, self.min_temp)

    def dequantize(self, inds):
        ## todo: need to be checked
        ## inds [B,T]
        bsz, tsz = inds.shape
        vars = self.embedding
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        indices = inds.reshape(bsz * tsz * self.groups, -1).flatten()
        vec = torch.zeros((bsz * tsz * self.groups, self.n_embeddings)).to(vars)
        hard_x = (
            vec.scatter_(-1, indices.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
        )
        x = hard_x.view(bsz * tsz, -1)
        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.n_embeddings, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        ### 2
        # vq_feat = (vars.squeeze(0).index_select(0, indices).reshape(bsz,tsz, -1))
        return x

    def get_extra_losses_type2(self, target_entropy):
        mae_loss = nn.L1Loss().to(self.code_entropy)
        target_entropy = torch.tensor(target_entropy).to(self.code_entropy)
        loss = 0.01 * mae_loss(self.code_entropy, target_entropy)
        return loss

    def get_extra_losses_type3(self):
        loss = self.config["lambda"] * self.code_entropy
        return loss

    def get_extra_losses_type4(self, target_entropy, fuzz_entropy):
        if self.code_entropy >= (target_entropy + fuzz_entropy) or self.code_entropy <= (target_entropy - fuzz_entropy):
            mae_loss = nn.L1Loss().to(self.code_entropy)
            loss = 0.01 * mae_loss(self.code_entropy, target_entropy)
        else:
            loss = torch.tensor(0.).to(self.code_entropy)
        return loss

    def forward(self, x, target_entropy=0, fuzz_entropy=0):
        output = {}
        _, M, D = self.embedding.size()  # 512 ,64
        weighted_code_entropy = torch.zeros(1,2,dtype=torch.float)
        self._eps = self._eps.to(x)      
        
        bsz, tsz, csz = x.shape
        x_flat = x.reshape(-1, csz)
        
        if 0 and self.config["use_random_sampled_decoder"] and not self.training:
            if self.config["random_sampling_type"] == 'uniform':
                distances_map = torch.rand(bsz * tsz * self.groups, M) # [BT, M] random within [0, 1)
                _, k = distances_map.max(-1) # [BT]
                hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups,-1))  # [BT, 1, M]
                distances_map = hard_x.view(bsz * tsz, -1)
                vars = self.embedding  # [1, M, D]
                if self.combine_groups:
                    vars = vars.repeat(1, self.groups, 1)
                output["quantization_inds"] = (
                    distances_map.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
                )
                distances_map = distances_map.unsqueeze(-1) * vars  # [BT, M, D]
                distances_map = distances_map.view(bsz * tsz, self.groups, self.n_embeddings, -1)
                distances_map = distances_map.sum(-2)  # [BT, D]
                quantized = distances_map.view(bsz, tsz, -1) # [B, T, D]
                output["quantized_feature"] = quantized
                return output

        # x_flat = self.weight_proj(x_flat) #BXT,C->BXT,vector_num
        if self.config["use_learnable_feat_cprs"]:
            x_flat = torch.where(x_flat > 0, (x_flat + self._eps) ** self.feat_power_cprs.to(x), -(-x_flat + self._eps) ** self.feat_power_cprs.to(x))
            #diff = x_flat.unsqueeze(dim=1).repeat(1, M, 1) - self.embedding   # [BT, M,  D]
            #distances = torch.sum((diff + self._eps) ** (2 * (self.feat_power_cprs.to(x) + self._eps)), dim=-1) # [BT, M]               
        #else:
        distances = torch.addmm(
            torch.sum(self.embedding.squeeze(0) ** 2, dim=1) + torch.sum(x_flat ** 2, dim=1, keepdim=True), x_flat,
            self.embedding.squeeze(0).t(), alpha=-2.0, beta=1.0)  # [BT, M]
        distances_map = torch.mul(self.alpha, distances)
        distances_map = distances_map.view(bsz * tsz * self.groups, -1) # [BT, M]

        _, k = distances_map.max(-1) # [BT]
        hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups,-1))  # [BT, 1, M]

        hard_probs = torch.mean(hard_x.float(), dim=0)
        output["code_perplexity"] = -torch.sum(hard_probs * torch.log2(hard_probs + 1e-10), dim=-1).squeeze(0)
        avg_probs = torch.softmax(distances_map.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        output["prob_perplexity"] = -torch.sum(avg_probs * torch.log2(avg_probs + 1e-10), dim=-1).squeeze(0)
        output["temp"] = self.curr_temp

        if self.training:
            # print(x.view(bsz * tsz * self.groups, -1).argmax(dim=-1))
            distances_map = F.gumbel_softmax(distances_map.float(), tau=self.curr_temp, hard=True).type_as(distances_map)
            # Compute entropy loss
            self.code_entropy = output["prob_perplexity"]  # rate control (entropy for current batch)
            # Weighted entropy regularization term
            weighted_code_entropy[:, 0] = self.code_entropy
            weighted_code_entropy[:, 1] = self.tau
            # overall entropy (for current epoch)
            self.entropy_avg_train.update(self.code_entropy.detach())
            avg_entropy = self.entropy_avg_train.avg
        else:
            distances_map = hard_x
            self.code_entropy = output["code_perplexity"]
            # Weighted entropy regularization term
            weighted_code_entropy[:, 0] = self.code_entropy
            weighted_code_entropy[:, 1] = self.tau
            # overall entropy
            self.entropy_avg_eval.update(self.code_entropy.detach())
            avg_entropy = self.entropy_avg_eval.avg

        distances_map = distances_map.view(bsz * tsz, -1)
        vars = self.embedding  # [1, M, D]
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        output["quantization_inds"] = (
            distances_map.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
        )       

        distances_map = distances_map.unsqueeze(-1) * vars  # [BT, M, D]
        distances_map = distances_map.view(bsz * tsz, self.groups, self.n_embeddings, -1)
        distances_map = distances_map.sum(-2)  # [BT, D]
        quantized = distances_map.view(bsz, tsz, -1) # [B, T, D]
        if self.config["use_learnable_feat_cprs"]:
            quantized = torch.where(quantized > 0, (quantized + self._eps) ** (1/(self.feat_power_cprs.to(x) + self._eps)), -(-quantized + self._eps) ** (1/(self.feat_power_cprs.to(x) + self._eps)))
        output["quantized_feature"] = quantized
        output["entropy"] = self.code_entropy # (entropy for current batch)
        output["entropy_avg"] = avg_entropy #  (entropy for current epoch)
        if self.config["use_entropy_loss"]:
            if self.config["entropy_loss_type"] == '1':
                output["entropy_loss"] = weighted_code_entropy
            elif self.config["entropy_loss_type"] == '2':
                output["entropy_loss"] = self.get_extra_losses_type2(target_entropy)
            elif self.config["entropy_loss_type"] == '3':
                output["entropy_loss"] = self.get_extra_losses_type3()
            elif self.config["entropy_loss_type"] == '4':
                output["entropy_loss"] = self.get_extra_losses_type4(target_entropy, fuzz_entropy)

        commitment_cost = F.mse_loss(x.detach(), quantized.detach())
        output["commitment_loss"] = commitment_cost ## just for show

        return output

class GumbelVectorQuantizer_Parallel(nn.Module):
    def __init__(self, config, input_dim, n_embeddings, groups, combine_groups=False):
        """Vector quantization using gumbel softmax
        Args:
            input_dim: input dimension (channels)
            n_embeddings: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = input_dim
        self.n_embeddings = n_embeddings 
        #complete_dim = int(config["sampling_rate"] * config["vq_in_dur"] * config["combineVQ_frames"])

        embedding_dim = input_dim // groups
        input_dim_per_group = input_dim // groups
        num_groups = groups if not combine_groups else 1
        assert(not combine_groups)
        
        self.embedding = nn.Parameter(torch.FloatTensor(num_groups, n_embeddings, embedding_dim))
        #nn.init.uniform_(self.embedding)
        nn.init.uniform_(self.embedding, a=-0.999, b=0.999)
        self.embedding_dim = embedding_dim

        self.max_temp, self.min_temp, self.temp_decay = config["temperature"]
        self.curr_temp = self.max_temp
        self.codebook_indices = None
        self.config = config

        self.entropy_avg_train = AverageMeter()
        self.entropy_avg_eval = AverageMeter()
        self.code_entropy = 0
        self.tau = 0
        self.tau2 = 0.5
        self.alpha = config["dist_to_logits_alpha"]      

        self._eps = torch.tensor(1e-7)          

    def temp_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates, self.min_temp)

    def dequantize(self, inds):
        ## todo: need to be checked
        ## inds [B,T]
        bsz, tsz, num_groups = inds.shape
        assert num_groups == self.groups
        vars = self.embedding
        if self.combine_groups:
            vars = vars.repeat(self.groups, 1, 1) # [group, M, D]

        indices = inds.reshape(bsz * tsz * self.groups, -1).flatten()
        vec = torch.zeros((bsz * tsz * self.groups, self.n_embeddings)).to(vars)
        hard_x = (
            vec.scatter_(-1, indices.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
        )
        x = hard_x.view(bsz * tsz, -1)
        x = x.unsqueeze(-1) * vars.reshape(1,-1,D)
        x = x.view(bsz * tsz, self.groups, self.n_embeddings, -1) # [BT, group, M, D]
        x = x.sum(-2) # [BT, group, D]
        x = x.view(bsz, tsz, -1)        

        ### 2
        # vq_feat = (vars.squeeze(0).index_select(0, indices).reshape(bsz, tsz, -1))
        return x

    def get_extra_losses_type2(self, target_entropy):
        mae_loss = nn.L1Loss().to(self.code_entropy)
        loss = 0
        for ii in range(self.groups):
            loss = loss + 0.01 * mae_loss(self.code_entropy[ii], target_entropy)
        return loss     
    
    def forward(self, x, target_entropy=0, fuzz_entropy=0, epo=None):
        output = {}        
        num_groups, M, D = self.embedding.size()
        bsz, tsz, csz = x.shape  # (B, T, group*D)
        self._eps = self._eps.to(x)        

        cur_x_flat = x.reshape(-1, self.groups, D)                          
          
        square_part = torch.sum(self.embedding.reshape(-1, D) ** 2, dim=1) + torch.sum(cur_x_flat ** 2, dim=2, keepdim=True).repeat(1,1,M).reshape(-1, self.groups*M) #(BT, group*M)
        cross_part = torch.matmul(cur_x_flat.unsqueeze(-2), self.embedding.permute(0,2,1).unsqueeze(0)).reshape(bsz*tsz,-1)            
        distances = square_part - 2*cross_part  #(BT, group*M)            
        
        distances_map = torch.mul(self.alpha, distances)
        distances_map = distances_map.view(bsz * tsz * self.groups, -1) # [BT*group, M]

        _, k = distances_map.max(-1) # [BT*group]
        hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)) # [BT, group, M]

        hard_probs = torch.mean(hard_x.float(), dim=0)
        output["code_perplexity"] = -torch.sum(hard_probs * torch.log2(hard_probs + 1e-10), dim=-1) #[group]
        avg_probs = torch.softmax(distances_map.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        output["prob_perplexity"] = -torch.sum(avg_probs * torch.log2(avg_probs + 1e-10), dim=-1) #[group]        
        
        # print(x.view(bsz * tsz * self.groups, -1).argmax(dim=-1))
        if self.config["use_defined_gumbelsoftmax"]:
            distances_map = gumbel_softmax(distances_map.float(), tau=self.curr_temp, hard=True).type_as(distances_map) # [BT*group, M]
        else:
            distances_map = F.gumbel_softmax(distances_map.float(), tau=self.curr_temp, hard=True).type_as(distances_map) # [BT*group, M]
        
        self.code_entropy = output["code_perplexity"]  # rate control (entropy for current batch)
        if self.training:
            # overall entropy (for current epoch)
            self.entropy_avg_train.update(torch.sum(self.code_entropy,dim=0).detach())
            avg_entropy = self.entropy_avg_train.avg
        else:
            # overall entropy
            self.entropy_avg_eval.update(torch.sum(self.code_entropy,dim=0).detach())
            avg_entropy = self.entropy_avg_eval.avg                

        distances_map = distances_map.view(bsz * tsz, -1) # [BT, group*M]
        vars = self.embedding   # [group, M, D]
        if self.combine_groups:
            vars = vars.repeat(self.groups, 1, 1) # [group, M, D]

        output["quantization_inds"] = (
            distances_map.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
        )

        output_temp = distances_map.unsqueeze(-1) * vars.reshape(1,-1,D) # [BT, group*M, D]
        output_temp = output_temp.view(bsz * tsz, self.groups, self.n_embeddings, -1) # [BT, group, M, D]
        output_temp = output_temp.sum(-2) # [BT, group, D]
            
        quantized = output_temp.view(bsz, tsz, -1)      
        
        output["temp"] = self.curr_temp
        output["quantized_feature"] = quantized
        output["entropy"] = self.code_entropy # (entropy for current batch)
        output["entropy_avg"] = avg_entropy #  (entropy for current epoch)
        if self.config["use_entropy_loss"]:
            output["entropy_loss"] = self.get_extra_losses_type2(target_entropy)         

        if self.config["use_vq_loss"]:
            output["commitment_loss"] = F.mse_loss(x, quantized.detach()) 
        else:
            output["commitment_loss"] = F.mse_loss(x.detach(), quantized.detach()) ## just for show            

        return output
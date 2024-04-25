import torch
import torch.nn as nn
import torch.nn.functional as F
from tfnet.utils.tools import AverageMeter
from .gumbel_max_pytorch import gumbel_softmax

class GumbelVectorQuantizer_Predictive(nn.Module):
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
        vq_dim = input_dim
        assert (
                vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        embedding_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        assert(not combine_groups)
        assert(groups == 1)

        self.embedding = nn.Parameter(torch.FloatTensor(1, num_groups * n_embeddings, embedding_dim))  # 1,1024,64        
        nn.init.uniform_(self.embedding)

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
        
        self.context_len = 7
        self.context_gen = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=self.context_len, stride=1, bias=False)
        
        if config["fuse_type"] == 'conv':
             self.fuse_conv1 = nn.Conv1d(2*embedding_dim, embedding_dim, kernel_size=1, stride=1, bias=False)
             self.nonlinearity1 = nn.PReLU(init=0.5) 
             self.fuse_conv2 = nn.Conv1d(2*embedding_dim, embedding_dim, kernel_size=1, stride=1, bias=False)
             self.nonlinearity2 = nn.PReLU(init=0.5) 

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

    def forward(self, x, target_entropy=0, fuzz_entropy=0, epo=None):
        output = {}
        _, M, D = self.embedding.size()  # 512 ,64        
        weighted_code_entropy = torch.zeros(1,2,dtype=torch.float)

        bsz, tsz, csz = x.shape  # (B, T, group*D)
        x_flat = x.reshape(-1, csz)        

        if self.training and (self.config["prediction_stage"] == '1'):
            ctx_in = F.pad(x[:, :-1, :].permute(0, 2, 1), (1+self.context_len-1, 0)).detach()
            uni_noise = torch.randn(ctx_in.size()).to(ctx_in)
            scale = torch.sqrt(torch.mean(torch.square(ctx_in), dim=(1,2), keepdim=True))
            ctx_in_noisy = ctx_in + 0.5*scale*uni_noise*(0.5**(epo/10.0)) # reduced per 10 epochs, scale-dependent           
            ctx_out = self.context_gen(ctx_in_noisy.detach())
            #ctx_out = self.context_gen(ctx_in)
            #ctx_out = ctx_in_noisy  # only use one previous feature
            context = ctx_out.permute(0, 2, 1)
            if self.config["fuse_type"] == 'conv':                
                merge_out = self.nonlinearity1(self.fuse_conv1(torch.cat((x.permute(0, 2, 1), ctx_out), dim=1)))
                cur_x = merge_out.permute(0, 2, 1)
            elif self.config["fuse_type"] == 'res':            
                cur_x = x - context   #[B, T, C]
            cur_x_flat = cur_x.reshape(-1, csz)
            
            distances = torch.addmm(torch.sum(self.embedding.squeeze(0) ** 2, dim=1) + torch.sum(cur_x_flat ** 2, dim=1, keepdim=True), cur_x_flat,
                self.embedding.squeeze(0).t(), alpha=-2.0, beta=1.0) # [BT, M]            
            distances_map = torch.mul(self.alpha, distances)
            distances_map = distances_map.view(bsz * tsz * self.groups, -1) # [BT*group, M/group]

            _, k = distances_map.max(-1) # [BT*group]
            hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups,-1)) # [BT, group, M/group]

            hard_probs = torch.mean(hard_x.float(), dim=0)
            output["code_perplexity"] = -torch.sum(hard_probs * torch.log2(hard_probs + 1e-10), dim=-1).squeeze(0)
            avg_probs = torch.softmax(distances_map.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
            output["prob_perplexity"] = -torch.sum(avg_probs * torch.log2(avg_probs + 1e-10), dim=-1).squeeze(0)
            
            
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

            distances_map = distances_map.view(bsz * tsz, -1) 
            vars = self.embedding   # [1, M, D]
            if self.combine_groups:
                vars = vars.repeat(1, self.groups, 1)

            output["quantization_inds"] = (
                distances_map.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
            )

            output_temp = distances_map.unsqueeze(-1) * vars # [BT, M, D]
            output_temp = output_temp.view(bsz * tsz, self.groups, self.n_embeddings, -1) # [BT, group, M, D/group]
            output_temp = output_temp.sum(-2) # [BT, group, D/group]
            output_q = output_temp.view(bsz, tsz, -1)
            if self.config["fuse_type"] == 'conv':
                merge_out = self.nonlinearity2(self.fuse_conv2(torch.cat((output_q.permute(0, 2, 1), ctx_out), dim=1)))
                quantized = merge_out.permute(0, 2, 1) # [B, T, D]
            elif self.config["fuse_type"] == 'res':
                quantized = output_q + context # [B, T, D]            
        else:
            context = torch.zeros(x.shape).to(x)
            quantized = torch.zeros(x.shape).to(x)
            #cur_x = torch.zeros(x.shape).to(x)
            #output_q = torch.zeros(x.shape).to(x)
            soft_prob_list = []
            hard_prob_list = []
            distances_map_list = []
            for ii in range(tsz):
                if self.config["fuse_type"] == 'conv':
                    merge_out = self.nonlinearity1(self.fuse_conv1(torch.cat((x[:,ii:ii+1,:].permute(0, 2, 1), context[:,ii:ii+1,:].permute(0, 2, 1)), dim=1)))
                    cur_x = merge_out.squeeze(dim=-1)                   
                elif self.config["fuse_type"] == 'res':
                    cur_x = x[:,ii,:] - context[:,ii,:]   #[B, D]
                #cur_x_flat = cur_x[:,ii,:]
                distances = torch.addmm(
                    torch.sum(self.embedding.squeeze(0) ** 2, dim=1) + torch.sum(cur_x ** 2, dim=1, keepdim=True), cur_x, self.embedding.squeeze(0).t(), alpha=-2.0, beta=1.0) # [BT, M]                    
                distances_map = torch.mul(self.alpha, distances)
                distances_map = distances_map.view(bsz * 1 * self.groups, -1)  # [BT*group, M/group]  

                _, k = distances_map.max(-1)
                hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * 1, self.groups,-1)) # [BT, group, M/group]
                hard_prob_list.append(hard_x.float())
                soft_prob_list.append(torch.softmax(distances_map.view(bsz * 1, self.groups, -1).float(), dim=-1))                
            
                if self.training:
                    distances_map = F.gumbel_softmax(distances_map.float(), tau=self.curr_temp, hard=True).type_as(distances_map)                    
                else:
                    distances_map = hard_x                    

                distances_map = distances_map.view(bsz * 1, -1)  # [BT, M]
                distances_map_list.append(distances_map)
                vars = self.embedding   # [1, M, D]
                if self.combine_groups:
                    vars = vars.repeat(1, self.groups, 1)                

                output_temp = distances_map.unsqueeze(-1) * vars # [BT, M, D]
                output_temp = output_temp.view(bsz * 1, self.groups, self.n_embeddings, -1) # [BT, group, M/group, D]
                output_temp = output_temp.sum(-2) # [BT, group, D/group]
                #output_q[:,ii:ii+1,:] = output_temp.view(bsz, 1, -1)
                output_q = output_temp.view(bsz, 1, -1)
                if self.config["fuse_type"] == 'conv':
                    merge_out = self.nonlinearity2(self.fuse_conv2(torch.cat((output_q.permute(0, 2, 1), context[:,ii:ii+1,:].permute(0, 2, 1)), dim=1)))
                    quantized[:,ii:ii+1,:] = merge_out.permute(0, 2, 1)
                elif self.config["fuse_type"] == 'res':
                    quantized[:,ii:ii+1,:] = output_q + context[:,ii:ii+1,:] # [B, T, D]
                if ii+1 < tsz:
                    if ii < self.context_len:
                        ctx_in = F.pad(quantized[:,0:ii,:].permute(0, 2, 1), (self.context_len-ii, 0))
                    else:
                        ctx_in = quantized[:,ii-self.context_len:ii,:].permute(0, 2, 1)
                    if self.config["use_BPTT"]:
                         ctx_out = self.context_gen(ctx_in)  #self.context_gen(ctx_in)
                        #ctx_out = ctx_in  # only use one previous feature
                    else:
                        ctx_out = self.context_gen(ctx_in.detach())  #self.context_gen(ctx_in)
                        #ctx_out = ctx_in.detach()  # only use one previous feature
                    context[:,ii+1:ii+2,:] = ctx_out.permute(0, 2, 1)                
            
            hard_probs = torch.mean(torch.cat(hard_prob_list, dim=0), dim=0)
            output["code_perplexity"] = -torch.sum(hard_probs * torch.log2(hard_probs + 1e-10), dim=-1).squeeze(0)
            avg_probs = torch.cat(soft_prob_list, dim=0).mean(dim=0)  # [group, M/group] 
            output["prob_perplexity"] = -torch.sum(avg_probs * torch.log2(avg_probs + 1e-10), dim=-1).squeeze(0)  # group must be 1
                
            self.code_entropy = output["code_perplexity"]  # rate control (entropy for current batch)
            # Weighted entropy regularization term
            weighted_code_entropy[:, 0] = self.code_entropy
            weighted_code_entropy[:, 1] = self.tau
            if self.training:
                # overall entropy (for current epoch)
                self.entropy_avg_train.update(self.code_entropy.detach())
                avg_entropy = self.entropy_avg_train.avg
            else:
                # overall entropy
                self.entropy_avg_eval.update(self.code_entropy.detach())
                avg_entropy = self.entropy_avg_eval.avg
                    
            output["quantization_inds"] = (
                    torch.stack(distances_map_list, dim=1).view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
                )       
        
        output["temp"] = self.curr_temp
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

        if self.config["use_vq_loss"]:
            output["commitment_loss"] = F.mse_loss(x, quantized.detach()) #F.mse_loss(cur_x, output_q.detach())
        else:
            output["commitment_loss"] = F.mse_loss(x.detach(), quantized.detach()) ## just for show
        if self.config["use_predictive_loss"]:
            output["predictive_loss"] = F.mse_loss(context, x.detach())
        else:
            output["predictive_loss"] = F.mse_loss(context.detach(), x.detach())

        return output


class GumbelVectorQuantizer_Predictive_Parallel(nn.Module):
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
        complete_dim = int(config["dft_size"] * config["hop_vqvae"] * config["combineVQ_frames"])
        vq_dim = complete_dim // 2 if (config["use_compressed_channels"] and config["fuse_type"] == 'conv') else input_dim
        if (config["use_compressed_channels"] and config["fuse_type"] == 'conv') and (vq_dim % groups):
            vq_dim = vq_dim + groups - (vq_dim % groups)
        assert (
                vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        embedding_dim = vq_dim // groups
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

        if config["prediction_type"] == 'conv':
            self.context_len = 7
            self.context_gen_list = nn.ModuleList(nn.Conv1d(input_dim_per_group, input_dim_per_group, kernel_size=self.context_len, stride=1, bias=False) for i in range(groups))
        elif config["prediction_type"] == 'conv-two':
            self.context_len_1 = 5
            self.context_gen_list_1 = nn.ModuleList(nn.Conv1d(input_dim_per_group, input_dim_per_group, kernel_size=self.context_len_1, stride=1, bias=True) for i in range(groups))
            self.context_nonlinearity1_list = nn.ModuleList(nn.PReLU(init=0.5) for i in range(groups))
            self.context_len_2 = 3
            self.context_gen_list_2 = nn.ModuleList(nn.Conv1d(input_dim_per_group, input_dim_per_group, kernel_size=self.context_len_2, stride=1, bias=False) for i in range(groups))
            self.context_len = 7
        elif config["prediction_type"] == 'gru':
            self.gru_context = nn.ModuleList(nn.GRU(input_dim_per_group, input_dim_per_group, num_layers=1, bias=True, batch_first=True) for i in range(groups))
            self.h0_context = nn.Parameter(torch.FloatTensor(1, 1, vq_dim).zero_())
            self.context_len = 1
        elif self.config["prediction_type"] == 'adaptive':
            self.context_len = 10
            self.softmax = torch.nn.Softmax(dim=-1)
            if self.config["use_learnable_alpha"]:
                self.adaptive_alpha = nn.Parameter(torch.FloatTensor(1).zero_().add(1), requires_grad=True)
            if self.config["use_affine_adaptive"]:
                self.context_gen_list = nn.ModuleList(nn.Conv1d(input_dim_per_group, input_dim_per_group, kernel_size=1, stride=1, bias=True) for i in range(groups))

        if config["fuse_type"] == 'conv':
            self.fuse_conv1_list = nn.ModuleList(nn.Conv1d(2*input_dim_per_group, embedding_dim, kernel_size=1, stride=1, bias=False) for i in range(groups)) # before vq
            self.nonlinearity1_list = nn.ModuleList(nn.PReLU(init=0.5) for i in range(groups))
            self.fuse_conv2_list = nn.ModuleList(nn.Conv1d(input_dim_per_group+embedding_dim, input_dim_per_group, kernel_size=1, stride=1, bias=False) for i in range(groups)) # after vq
            self.nonlinearity2_list = nn.ModuleList(nn.PReLU(init=0.5) for i in range(groups))

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
        loss = 0
        for ii in range(self.groups):
            loss = loss + 0.01 * mae_loss(self.code_entropy[ii], target_entropy)
        return loss

    # def get_extra_losses_type3(self):
        # loss = self.config["lambda"] * self.code_entropy
        # return loss

    # def get_extra_losses_type4(self, target_entropy, fuzz_entropy):
        # if self.code_entropy >= (target_entropy + fuzz_entropy) or self.code_entropy <= (target_entropy - fuzz_entropy):
            # mae_loss = nn.L1Loss().to(self.code_entropy)
            # loss = 0.01 * mae_loss(self.code_entropy, target_entropy)
        # else:
            # loss = torch.tensor(0.).to(self.code_entropy)
        # return loss
    
    def _gru_init_state(self, n):
        if not torch._C._get_tracing_state():
            return self.h0_context.expand(-1, n, -1).contiguous()
        else:
            return self.h0_context.expand(self.h0_context.size(0), n, self.h0_context.size(2)).contiguous()
    
    def predictor(self, x, state=None): # (B,C,T)
        bsz, csz, tsz = x.shape
        ctx_out_list = []
        state_out = None
        if self.config["prediction_type"] == 'conv':
            ctx_in_list = torch.chunk(x, self.groups, dim=1)
            for ii in range(self.groups):
                ctx_out_list.append(self.context_gen_list[ii](ctx_in_list[ii]))
        elif self.config["prediction_type"] == 'conv-two':
            ctx_in_list = torch.chunk(x, self.groups, dim=1)
            for ii in range(self.groups):
                ctx_out_1 = torch.tanh(self.context_gen_list_1[ii](ctx_in_list[ii])) if self.config["use_tanh_features"] else self.context_nonlinearity1_list[ii](self.context_gen_list_1[ii](ctx_in_list[ii]))
                ctx_out_list.append(self.context_gen_list_2[ii](ctx_out_1))
        elif self.config["prediction_type"] == 'gru':
            x_in = x.permute(0, 2, 1)
            ctx_in_list = torch.chunk(x_in, self.groups, dim=2)  # (B,T,C)
            state_in_list = torch.chunk(state, self.groups, dim=2) # (1,B,C)
            state_out_list = []
            for ii in range(self.groups):  
                rnn_out, h_n = self.gru_context[ii](ctx_in_list[ii], state_in_list[ii].contiguous())
                ctx_out_list.append(rnn_out.permute(0, 2, 1))  # (B,C,T)
                state_out_list.append(h_n) # (1,B,C)
            state_out = torch.cat(state_out_list, dim=2)
        elif self.config["prediction_type"] == 'adaptive':
            ctx_in_list = torch.chunk(x, self.groups, dim=1)    
            for ii in range(self.groups):
                if self.config["use_affine_adaptive"]:
                    query = self.context_gen_list[ii](ctx_in_list[ii][:,:,-1].unsqueeze(-1)) #(B, C, 1)
                    key = self.context_gen_list[ii](ctx_in_list[ii][:,:,:-1]) #(B, C, T-1)
                    value = self.context_gen_list[ii](ctx_in_list[ii][:,:,1:]) #(B, C, T-1)
                else:   
                    query = ctx_in_list[ii][:,:,-1].unsqueeze(-1) #(B, C, 1)
                    key = ctx_in_list[ii][:,:,:-1] #(B, C, T-1)
                    value = ctx_in_list[ii][:,:,1:] #(B, C, T-1)
                ssa_mat = torch.matmul(query.permute(0,2,1), key) # [B, 1, T-1]
                if self.config["use_learnable_alpha"]:
                    ssa_softmax = self.softmax(self.adaptive_alpha*ssa_mat/(csz**0.5)) # [B, 1, T-1]
                else:
                    ssa_softmax = self.softmax(ssa_mat/(csz**0.5)) # [B, 1, T-1]
                att_output = torch.matmul(ssa_softmax, value.permute(0,2,1))  # [B, 1, C]
                ctx_out_list.append(att_output.permute(0,2,1)) # [B, C, 1]
        output = torch.cat(ctx_out_list, dim=1)
        return output, state_out

    def forward(self, x, target_entropy=0, fuzz_entropy=0, ar_signal=None, epo=None):
        output = {}
        num_groups, M, D = self.embedding.size()
        bsz, tsz, csz = x.shape  # (B, T, group*D)
        self._eps = self._eps.to(x)

        if self.training and (ar_signal is not None):
            ctx_in = F.pad(ar_signal[:, :-1, :].permute(0, 2, 1), (1+self.context_len-1, 0)).detach()
            if self.config["prediction_stage"] == '1':
                uni_noise = torch.randn(ctx_in.size()).to(ctx_in)
                scale = torch.sqrt(torch.mean(torch.square(ctx_in), dim=(1,2), keepdim=True))
                ctx_in_noisy = ctx_in + 0.5*scale*uni_noise*(0.5**(epo/10.0)) # reduced per 10 epochs, scale-dependent
            else:
                ctx_in_noisy = ctx_in

            if self.config["prediction_type"] == 'gru':
                ctx_out, _ = self.predictor(ctx_in_noisy, state=self._gru_init_state(bsz))
            else:
                ctx_out, _ = self.predictor(ctx_in_noisy)
            context = ctx_out.permute(0, 2, 1)
            if self.config["fuse_type"] == 'conv':
                x_list = torch.chunk(x.permute(0, 2, 1), self.groups, dim=1)
                ctx_out_list = torch.chunk(context.permute(0, 2, 1), self.groups, dim=1)
                cur_x_list = []
                for ii in range(self.groups):    
                    merge_out = torch.tanh(self.fuse_conv1_list[ii](torch.cat((x_list[ii], ctx_out_list[ii]), dim=1))) if self.config["use_tanh_before_vq"] else self.nonlinearity1_list[ii](self.fuse_conv1_list[ii](torch.cat((x_list[ii], ctx_out_list[ii]), dim=1)))
                    cur_x_list.append(merge_out)
                cur_x = torch.cat(cur_x_list, dim=1).permute(0, 2, 1)
            elif self.config["fuse_type"] == 'res':  
                cur_x = x - context   #[B, T, C]
            cur_x_flat = cur_x.reshape(-1, self.groups, D)
            vq_in = cur_x

            if self.config["use_learnable_feat_cprs"]:
                cur_x_flat = torch.where(cur_x_flat > 0, (cur_x_flat + self._eps) ** self.feat_power_cprs, -(-cur_x_flat + self._eps) ** self.feat_power_cprs)
                #cur_x_flat = torch.tanh(cur_x_flat * self.feat_power_cprs)#.to(x))

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
            # Compute entropy loss
            self.code_entropy = output["prob_perplexity"]  # rate control (entropy for current batch)
            # overall entropy (for current epoch)
            self.entropy_avg_train.update(torch.sum(self.code_entropy,dim=0).detach())
            avg_entropy = self.entropy_avg_train.avg

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
            if self.config["use_learnable_feat_cprs"]:
                output_temp = torch.where(output_temp > 0, (output_temp + self._eps) ** (1/(self.feat_power_cprs + self._eps)), -(-output_temp + self._eps) ** (1/(self.feat_power_cprs + self._eps)))
                #output_temp = torch.atanh(output_temp) / (self.feat_power_cprs)#.to(x) + self._eps)
  
            output_q = output_temp.view(bsz, tsz, -1)
            vq_out = output_q
            if self.config["fuse_type"] == 'conv':
                merge_out_list = []
                output_q_list = torch.chunk(output_q.permute(0, 2, 1), self.groups, dim=1)
                for ii in range(self.groups):
                    merge_out_list.append(torch.tanh(self.fuse_conv2_list[ii](torch.cat((output_q_list[ii], ctx_out_list[ii]), dim=1))) if self.config["use_tanh_features"] else self.nonlinearity2_list[ii](self.fuse_conv2_list[ii](torch.cat((output_q_list[ii], ctx_out_list[ii]), dim=1))))
                merge_out = torch.cat(merge_out_list, dim=1)
                quantized = merge_out.permute(0, 2, 1) # [B, T, D]
            elif self.config["fuse_type"] == 'res':
                quantized = output_q + context # [B, T, D]
        else:
            context = torch.zeros(x.shape).to(x)
            quantized = torch.zeros(x.shape).to(x)
            soft_prob_list = []
            hard_prob_list = []
            distances_map_list = []
            aggregator = torch.zeros((tsz+1, bsz, csz)).contiguous().to(x)
            vq_in_list = []
            vq_out_list = []
            if 0 and self.config["use_random_sampled_decoder"]:             
                if self.config["random_sampling_type"] == 'uniform':
                    for ii in range(tsz):
                        distances_map = torch.rand(bsz, self.groups*M) # #(B, group*M) random within [0, 1)
                        distances_map = distances_map.view(bsz * 1 * self.groups, -1)  # [B*group, M]
                        _, k = distances_map.max(-1)
                        hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * 1, self.groups,-1)) # [B, group, M]
                        distances_map = hard_x.view(bsz * 1, -1)  # [B, group*M]
                        distances_map_list.append(distances_map)
                        vars = self.embedding  # [1, M, D]
                        if self.combine_groups:
                            vars = vars.repeat(self.groups, 1, 1)            

                        output_temp = distances_map.unsqueeze(-1) * vars.reshape(1,-1,D) # [B, group*M, D]
                        output_temp = output_temp.view(bsz * 1, self.groups, self.n_embeddings, -1) # [B, group, M, D]
                        output_temp = output_temp.sum(-2) # [B, group, D]

                        output_q = output_temp.view(bsz, 1, -1)
                        vq_out_list.append(output_q)
                        if self.config["fuse_type"] == 'conv':
                            merge_out_list = []
                            output_q_list = torch.chunk(output_q.permute(0, 2, 1), self.groups, dim=1)
                            context_list = torch.chunk(context[:,ii:ii+1,:].permute(0, 2, 1), self.groups, dim=1)
                            for gg in range(self.groups):
                                merge_out_list.append(self.nonlinearity2_list[gg](self.fuse_conv2_list[gg](torch.cat((output_q_list[gg], context_list[gg]), dim=1))))
                            merge_out = torch.cat(merge_out_list, dim=1)
                            quantized[:,ii:ii+1,:] = merge_out.permute(0, 2, 1) # [B, 1, D]
                        elif self.config["fuse_type"] == 'res':
                            quantized[:,ii:ii+1,:] = output_q + (context[:,ii:ii+1,:].detach() if self.config["use_context_detach"] else context[:,ii:ii+1,:]) # [B, 1, D]    
                        if ii+1 < tsz:      
                            if ii+1 < self.context_len:
                                ctx_in = F.pad(quantized[:,0:ii+1,:].permute(0, 2, 1), (self.context_len-ii-1, 0))
                            else:
                                ctx_in = quantized[:,ii+1-self.context_len:ii+1,:].permute(0, 2, 1)              
                            # if ii < self.context_len:
                                # ctx_in = F.pad(quantized[:,0:ii,:].permute(0, 2, 1), (self.context_len-ii, 0))
                            # else:
                                # ctx_in = quantized[:,ii-self.context_len:ii,:].permute(0, 2, 1)
                            if self.config["prediction_type"] == 'gru':
                                ctx_out, aggregator[ii+1:ii+2,:,:] = self.predictor(ctx_in if self.config["use_BPTT"] else ctx_in.detach(), state=aggregator[ii:ii+1,:,:])
                            else:
                                ctx_out, _ = self.predictor(ctx_in if self.config["use_BPTT"] else ctx_in.detach())
                            context[:,ii+1:ii+2,:] = ctx_out.permute(0, 2, 1)  
                            
                    vq_out = torch.cat(vq_out_list, dim=1)                            
                    output["quantization_inds"] = (
                            torch.stack(distances_map_list, dim=1).view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
                        )      
                    output["quantized_feature"] = quantized
                    return output
                            
            for ii in range(tsz):
                if self.config["fuse_type"] == 'conv':
                    x_list = torch.chunk(x[:,ii:ii+1,:].permute(0, 2, 1), self.groups, dim=1)
                    if self.config["use_context_detach"]:
                        context_list = torch.chunk(context[:,ii:ii+1,:].detach().permute(0, 2, 1), self.groups, dim=1) 
                    else:
                        context_list = torch.chunk(context[:,ii:ii+1,:].permute(0, 2, 1), self.groups, dim=1) 
                    cur_x_list = []
                    for gg in range(self.groups):            
                        merge_out = torch.tanh(self.fuse_conv1_list[gg](torch.cat((x_list[gg], context_list[gg]), dim=1))) if self.config["use_tanh_before_vq"] else self.nonlinearity1_list[gg](self.fuse_conv1_list[gg](torch.cat((x_list[gg], context_list[gg]), dim=1)))
                        cur_x_list.append(merge_out)
                    cur_x = torch.cat(cur_x_list, dim=1).permute(0, 2, 1)#torch.cat(cur_x_list, dim=1).squeeze(dim=-1) #(B, group*D)                                 
                elif self.config["fuse_type"] == 'res':
                    cur_x = x[:,ii:ii+1,:] - (context[:,ii:ii+1,:].detach() if self.config["use_context_detach"] else context[:,ii:ii+1,:])   #[B, group*D]                
                cur_x_flat = cur_x.reshape(-1, self.groups, D)
                vq_in_list.append(cur_x)
                
                if self.config["use_learnable_feat_cprs"]:
                    cur_x_flat = torch.where(cur_x_flat > 0, (cur_x_flat + self._eps) ** self.feat_power_cprs, -(-cur_x_flat + self._eps) ** self.feat_power_cprs)
                    #cur_x_flat = torch.tanh(cur_x_flat * self.feat_power_cprs)#.to(x))
                square_part = torch.sum(self.embedding.reshape(-1, D) ** 2, dim=1) + torch.sum(cur_x_flat ** 2, dim=2, keepdim=True).repeat(1,1,M).reshape(-1, self.groups*M) #(B, group*M)
                cross_part = torch.matmul(cur_x_flat.unsqueeze(-2), self.embedding.permute(0,2,1).unsqueeze(0)).reshape(bsz,-1)                
                distances = square_part - 2*cross_part  #(B, group*M)
                distances_map = torch.mul(self.alpha, distances)
                distances_map = distances_map.view(bsz * 1 * self.groups, -1)  # [B*group, M]  

                _, k = distances_map.max(-1)
                hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * 1, self.groups,-1)) # [B, group, M]
                hard_prob_list.append(hard_x.float())
                soft_prob_list.append(torch.softmax(distances_map.view(bsz * 1, self.groups, -1).float(), dim=-1))                
            
                if self.training:
                    if self.config["use_defined_gumbelsoftmax"]:
                        distances_map = gumbel_softmax(distances_map.float(), tau=self.curr_temp, hard=True).type_as(distances_map) 
                    else:
                        distances_map = F.gumbel_softmax(distances_map.float(), tau=self.curr_temp, hard=True).type_as(distances_map)                    
                else:
                    distances_map = hard_x                    

                distances_map = distances_map.view(bsz * 1, -1)  # [B, group*M]
                distances_map_list.append(distances_map)
                vars = self.embedding   # [group, M, D]
                if self.combine_groups:
                    vars = vars.repeat(self.groups, 1, 1)                

                output_temp = distances_map.unsqueeze(-1) * vars.reshape(1,-1,D) # [B, group*M, D]
                output_temp = output_temp.view(bsz * 1, self.groups, self.n_embeddings, -1) # [B, group, M, D]
                output_temp = output_temp.sum(-2) # [B, group, D]
                if self.config["use_learnable_feat_cprs"]:
                    output_temp = torch.where(output_temp > 0, (output_temp + self._eps) ** (1/(self.feat_power_cprs + self._eps)), -(-output_temp + self._eps) ** (1/(self.feat_power_cprs + self._eps)))
                    #output_temp = torch.atanh(output_temp) / (self.feat_power_cprs)#.to(x) + self._eps)
                    
                output_q = output_temp.view(bsz, 1, -1)
                vq_out_list.append(output_q)
                if self.config["fuse_type"] == 'conv':
                    merge_out_list = []
                    output_q_list = torch.chunk(output_q.permute(0, 2, 1), self.groups, dim=1)
                    if self.config["use_context_detach"]:
                        context_list = torch.chunk(context[:,ii:ii+1,:].detach().permute(0, 2, 1), self.groups, dim=1)
                    else:
                        context_list = torch.chunk(context[:,ii:ii+1,:].permute(0, 2, 1), self.groups, dim=1)
                    for gg in range(self.groups):
                        merge_out_list.append(torch.tanh(self.fuse_conv2_list[gg](torch.cat((output_q_list[gg], context_list[gg]), dim=1))) if self.config["use_tanh_features"] else self.nonlinearity2_list[gg](self.fuse_conv2_list[gg](torch.cat((output_q_list[gg], context_list[gg]), dim=1))))
                    merge_out = torch.cat(merge_out_list, dim=1)
                    quantized[:,ii:ii+1,:] = merge_out.permute(0, 2, 1) # [B, 1, D]                
                elif self.config["fuse_type"] == 'res':
                    quantized[:,ii:ii+1,:] = output_q + (context[:,ii:ii+1,:].detach() if self.config["use_context_detach"] else context[:,ii:ii+1,:]) # [B, 1, D]                    
                if ii+1 < tsz:                
                    if ii+1 < self.context_len:
                        ctx_in = F.pad(quantized[:,0:ii+1,:].permute(0, 2, 1), (self.context_len-ii-1, 0))
                    else:
                        ctx_in = quantized[:,ii+1-self.context_len:ii+1,:].permute(0, 2, 1)                        
                    # if ii < self.context_len:
                        # ctx_in = F.pad(quantized[:,0:ii,:].permute(0, 2, 1), (self.context_len-ii, 0))
                    # else:
                        # ctx_in = quantized[:,ii-self.context_len:ii,:].permute(0, 2, 1)
                    if self.config["prediction_type"] == 'gru':
                        ctx_out, aggregator[ii+1:ii+2,:,:] = self.predictor(ctx_in if self.config["use_BPTT"] else ctx_in.detach(), state=aggregator[ii:ii+1,:,:])
                    else:
                        ctx_out, _ = self.predictor(ctx_in if self.config["use_BPTT"] else ctx_in.detach())
                    context[:,ii+1:ii+2,:] = ctx_out.permute(0, 2, 1)                                   
            
            vq_in = torch.cat(vq_in_list, dim=1)
            vq_out = torch.cat(vq_out_list, dim=1)
            hard_probs = torch.mean(torch.cat(hard_prob_list, dim=0), dim=0) # [group, M]
            output["code_perplexity"] = -torch.sum(hard_probs * torch.log2(hard_probs + 1e-10), dim=-1) #[group]
            avg_probs = torch.cat(soft_prob_list, dim=0).mean(dim=0)  # [group, M] 
            output["prob_perplexity"] = -torch.sum(avg_probs * torch.log2(avg_probs + 1e-10), dim=-1) #[group]
                
            self.code_entropy = output["code_perplexity"]  # rate control (entropy for current batch)
            if self.training:
                # overall entropy (for current epoch)
                self.entropy_avg_train.update(torch.sum(self.code_entropy,dim=0).detach())
                avg_entropy = self.entropy_avg_train.avg
            else:
                # overall entropy
                self.entropy_avg_eval.update(torch.sum(self.code_entropy,dim=0).detach())
                avg_entropy = self.entropy_avg_eval.avg
                    
            output["quantization_inds"] = (
                    torch.stack(distances_map_list, dim=1).view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
                )  
            if 0:
                # calculate temporal correlation coefficient
                mean = torch.mean(vq_in[:,:-1,:], dim=1)
                cross_correlation = torch.sum(torch.matmul(vq_in[:,:-1,:].reshape(tsz-1, 1, self.groups*D)-mean, vq_in[:,1:,:].reshape(tsz-1, self.groups*D, 1)-mean.reshape(-1,1)),dim=0)
                self_correlation = torch.sum(torch.matmul(vq_in[:,:-1,:].reshape(-1, 1, self.groups*D)-mean, vq_in[:,:-1,:].reshape(-1, self.groups*D, 1)-mean.reshape(-1,1)),dim=0)
                corr_coef = cross_correlation/self_correlation
                print('vq_in coeff: {}, cross: {}, self: {}'.format(corr_coef, cross_correlation, self_correlation))          

                mean = torch.mean(x[:,:-1,:], dim=1)
                cross_correlation = torch.sum(torch.matmul(x[:,:-1,:].reshape(tsz-1, 1, csz)-mean, x[:,1:,:].reshape(tsz-1, csz, 1)-mean.reshape(-1,1)),dim=0)
                self_correlation = torch.sum(torch.matmul(x[:,:-1,:].reshape(-1, 1, csz)-mean, x[:,:-1,:].reshape(-1, csz, 1)-mean.reshape(-1,1)),dim=0)
                corr_coef = cross_correlation/self_correlation
                print('encoder feature coeff: {}, cross: {}, self: {}'.format(corr_coef, cross_correlation, self_correlation))                    
        
        output["temp"] = self.curr_temp
        output["quantized_feature"] = quantized
        output["entropy"] = self.code_entropy # (entropy for current batch)
        output["entropy_avg"] = avg_entropy #  (entropy for current epoch)
        if self.config["use_entropy_loss"]:
            output["entropy_loss"] = self.get_extra_losses_type2(target_entropy)         

        if self.config["use_vq_loss"]:
            output["commitment_loss"] = F.mse_loss(x, quantized.detach()) #F.mse_loss(vq_in, vq_out.detach())
        else:
            output["commitment_loss"] = F.mse_loss(x.detach(), quantized.detach()) ## just for show
            
        if self.config["use_predictive_loss"]:
            output["predictive_loss"] = F.mse_loss(context, x.detach() if self.config["use_detached_pred_loss"] else x)
        else:
            output["predictive_loss"] = F.mse_loss(context.detach(), x.detach())            
        
        if self.config["use_sparse_loss"]:
            output["feature_sparse_loss"] = F.l1_loss(vq_in, torch.zeros_like(vq_in))
        else:
            output["feature_sparse_loss"] = F.l1_loss(vq_in.detach(), torch.zeros_like(vq_in))

        return output


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
h_img=0
w_img=0
seq_len = 10
def generate_ref_points(width: int, height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'layer': nn.LayerNorm,
        }.get(normalization, None)

        if normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, elementwise_affine=True)
        else:
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):

            output = self.normalizer(input.view(-1, input.size(-1))).view(*input.size())

            return output  # self.normalizer(input.view(-1, input.size(-1))).view(*input.size())

        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)

        else:
            assert self.normalizer is None, 'Unknown normalizer type'
            return input


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, input_dim, embed_dim, v_dim=None, k_dim=None, dropout=0.1, K_num=3):

        super(MultiHeadAttention, self).__init__()

        if v_dim is None:
            v_dim = embed_dim // n_head

        if k_dim is None:
            k_dim = v_dim

        self.n_head = n_head
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.v_dim = v_dim
        self.k_dim = k_dim

        self.dropout = dropout

        self.norm_factor = 1 / math.sqrt(k_dim)

        C = 1280;

        # C=8
        # K = 4;
        M = 8;
        L = 1;
        S = seq_len

        self.C_v = v_dim
        self.M = M
        self.L = L
        self.K_num = K_num

        self.Q = nn.Parameter(torch.Tensor(n_head, input_dim, k_dim))
        self.K = nn.Parameter(torch.Tensor(n_head, input_dim, k_dim))
        self.V = nn.Parameter(torch.Tensor(n_head, input_dim, v_dim*self.K_num))

        self.W = nn.Parameter(torch.Tensor(n_head, v_dim, embed_dim))



        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.W_prim = nn.Linear(embed_dim, embed_dim)
        # self.delta_proj = nn.Linear(C, 2 * M * L * K) # delta p_q 2 *L* M * K
        self.delta_proj = nn.Linear(v_dim*S, 2  * K_num* seq_len)  # delta p_q 2 *L* M * K
        self.Attention_projection = nn.Linear(C, 2*M,K_num*L)


        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

        torch.nn.init.constant_(self.delta_proj.weight, 0.0)
        torch.nn.init.constant_(self.Attention_projection.weight, 0.0)

        torch.nn.init.constant_(self.Attention_projection.bias, 1 / (self.L * self.K_num))

        def init_xy(bias, x, y):

            torch.nn.init.constant_(bias, float(x))
            torch.nn.init.constant_(bias, float(y))
        # caution: offset layout will be  M, L, K, 2

        bias = self.delta_proj.bias
        list_num = [-1,0,1]
        # list_num = [0]
        for i in range(len(bias)):
            init_xy(bias[i], x = random.choice(list_num), y = random.choice(list_num))







    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        # height = int(batch_size ** 0.5)
        # width = int(batch_size ** 0.5)
        height = h_img
        width = w_img
        batch_n = batch_size//(width*height)


        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = h.contiguous().view(-1, input_dim)

        shp = (self.n_head, batch_size, graph_size, -1)
        shp_q = (self.n_head, batch_size, n_query, -1)

        # Calculate queries
        q = torch.matmul(q_flat, self.Q).view(shp_q)  # n_head, batch, query, k_dim // M, H,W, S, C_v

        # Calculate keys and values
        k = torch.matmul(h_flat, self.K).view(shp)  # n_head, batch, graph_size, k_dim
        # v = torch.matmul(h_flat, self.V).view(shp)  # n_head, batch, graph_size, v_dim
        shp_v = (shp[0], shp[1], shp[2]*self.K_num, shp[3])
        v = torch.matmul(h_flat, self.V).view(shp_v)

        #deformable DETR

        # z_q = self.q_proj(z_q)
        q = q.permute(1, 2, 3, 0).contiguous()
        q = q.view(self.n_head,batch_size,-1)
        deltas = self.delta_proj(q)  # B, H, W, 2MLK
        # deltas = deltas.view(B, H, W, self.M, -1)  # B, H, W, M, 2LK
        # deltas = deltas.view(B, H, W, self.M, self.L, self.K_num, 2)  # batch , H, W, M, L, K, 2
        deltas = deltas.view(batch_n* self.n_head, height,width, n_query,self.K_num, 2) # B, M, H, W, S, K, 2

        # deltas = deltas.permute(0, 3, 4, 5, 1, 2, 6).contiguous()  # Batch, M, L, K, H, W, 2
        # deltas = deltas.permute(0, 1, 2,3, )

        # deltas = deltas.view(B * self.M, self.L, self.K_num, H, W, 2)  # Bacth * M, L, K, H, W, 2

        ref_point = generate_ref_points(width, height)  # H,W,2
        # ref_point = torch.stack((ref_point, ref_point), 2).permute(3,0,1,2) # B, H, W, 2

        # ref_point = ref_point.type_as(torch.IntTensor())
        # ref_point = ref_point.type_as(src[0])  #???????????????????????
        bs =1
        ref_point = ref_point.unsqueeze(0).repeat(bs, 1, 1, 1).cuda(device=q.device)


        ref_point = ref_point.repeat(batch_n*self.M, 1, 1, 1) # BM, H, W, 2 # repeat M points for every attention head

        sampled_features = self.compute_sampling(k, ref_point, deltas, seq_len=n_query, h=height, w=width)


        sampled_features_scaled = sampled_features


        sampled_features_scaled = sampled_features_scaled.permute(0,2,3,4,1)
        sampled_features_scaled = sampled_features_scaled.reshape(self.M,batch_size,self.K_num * n_query,self.v_dim)
        k = sampled_features_scaled
        q = q.view(self.M,batch_size,n_query,self.v_dim)



        #normal

        qk = self.norm_factor * torch.matmul(q, k.transpose(2, 3))  # n_head, batch, query, graph_size

        attn = torch.softmax(qk, dim=-1)

        attn_mean = attn.mean(0)
        # attn_map = torch.stack([attn_mean[:,i].sum(-1) for i in range(graph_size)], 1) # bhw x seqlen
        # attn_map = torch.stack(attn_mean[:, 0], 1)
        attn_map = attn_mean[:, 0]  # method2

        head = torch.matmul(attn, v)  # n_head, batch, query, v_dim

        head_flat = head.permute(1, 2, 0, 3).contiguous().view(-1,
                                                               self.n_head * self.v_dim)  # (batch x query), (n_head x v_dim)

        w_flat = self.W.view(-1, self.embed_dim)  # (n_head x v_dim), embedded_dim

        out = torch.mm(head_flat, w_flat).view(batch_size, n_query, self.embed_dim)

        if self.training:
            dropout_mask = out.new(out.size(0), out.size(2)).bernoulli_(1 - self.dropout).div(1 - self.dropout)
            out = out * dropout_mask.unsqueeze(1).expand_as(out)

        return out, attn_map

    def compute_sampling(self, W_prim_x, phi_p_q, deltas, seq_len, h, w):
        offseted_features = []
        for k in range(self.K_num):  # for K points
            for s in range(seq_len):  # for S points
                phi_p_q_plus_deltas = phi_p_q + deltas[:, :, :, s, k,:].squeeze(0)  # p_q + delta p_mqk


                vgrid_x = F.normalize(phi_p_q_plus_deltas[:, :, :, 0])
                vgrid_y = F.normalize(phi_p_q_plus_deltas[:, :, :, 1])

                vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)  # stack the
                temp = W_prim_x.view(phi_p_q.shape[0], h,w,seq_len, self.v_dim)
                temp = temp.permute(0,4,1,2,3)
                sampled = F.grid_sample(temp[:,:,:,:,s], vgrid_scaled, mode='bilinear', padding_mode='zeros')

                offseted_features.append(sampled)
        return torch.stack(offseted_features, dim=4)






class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, n_head, embed_dim, feed_forward_hidden=512, dropout=0.1, normalization='layer', K_num=3):
        super(MultiHeadAttentionLayer, self).__init__()

        self.K_num = K_num
        self.MHA = MultiHeadAttention(
            n_head,
            input_dim=embed_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            K_num = K_num,
        )

        self.FF = nn.Sequential(
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

    def forward(self, inputs):
        outputs, attn_map = self.MHA(inputs)
        attn_map_out = attn_map[:,0:seq_len]
        for i in range(self.K_num-1):
            attn_map_out = attn_map_out + attn_map[:,seq_len*(i+1):seq_len*(i+1)+seq_len]
        # attn_map = (attn_map[:,0:10] + attn_map[:,10:20] + attn_map[:,20:30])/3
        attn_map_out = attn_map_out / self.K_num
        outputs = outputs + inputs  # skip connection

        outputs = self.FF(outputs)

        return outputs, attn_map_out


class DETR(nn.Module):
    def __init__(self, n_defect):
        super(DETR, self).__init__()

        self.n_head = 8
        self.n_feat = 1280
        self.n_defect = n_defect
        self.K_num = 3
        self.layers = nn.ModuleList(
            [MultiHeadAttentionLayer(self.n_head, self.n_feat, 128, 0, 'layer', self.K_num) for _ in range(2)]
        )

        self.lin_layer1 = nn.Sequential(
            nn.Conv2d(self.n_feat, 1000, kernel_size=1),
        )

        self.pool_layer = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.lin_layer2 = torch.nn.Linear(10,2)

    def forward(self, inputs):

        # inputs: BSDHW
        batch_size, graph_size, feat_size, h, w = inputs.size()
        global h_img
        global w_img
        h_img = h
        w_img = w
        inputs_flatten = inputs.flatten(3).permute(0, 3, 1, 2)  # BSD HW
        inputs_flatten = inputs_flatten.contiguous().view(-1, graph_size, feat_size)

        # add positional encoding
        # generate pe vector
        pe = torch.zeros(graph_size, feat_size)
        position = torch.arange(0, graph_size).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_size, 2).float() * (-math.log(10000.0) / feat_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.to(device=inputs.device)

        inputs_flatten = inputs_flatten + pe
        attn_map = []
        for i in range(len(self.layers) - 1):
            # inputs_flatten, _ = self.layers[i](inputs_flatten)
            inputs_flatten, attn = self.layers[i](inputs_flatten)
            attn_map.append(attn)
        feats_flatten, attn = self.layers[-1](inputs_flatten)

        # calculate attn average


        attn_map.append(attn)
        # attn_maps = torch.cat(attn_map, 1)

        attn_maps = torch.stack(attn_map)  # 12 * B * H * N * N
        attn_maps = torch.sum(attn_maps, dim=0)  # 12 * B * N * N

        attn_maps = attn_maps.view(batch_size, h, w, graph_size).permute(0, 3, 1, 2)
        # attn_maps: batch x seqlen x h x w

        feats = feats_flatten.view(batch_size, h, w, graph_size, feat_size)
        feats = feats.permute(0, 3, 4, 1, 2).contiguous().view(-1, feat_size, h, w)

        maps = self.lin_layer1(feats)
        outputs = self.pool_layer(maps)
        # outputs = outputs.permute(3,0,1,2)
        # outputs = self.lin_layer2(outputs.view(batch_size, -1))
        outputs = outputs.squeeze()


        # outputs = outputs.view(batch_size, graph_size, self.n_defect)
        # maps = maps.view(batch_size, graph_size, *maps.size()[1:])

        # pred_digit = self.lin_layer(outputs.view(batch_size, graph_size))
        # return maps, outputs, attn_maps, pred_digit

        return outputs
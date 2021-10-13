"""
TRTR Transformer class.

Copy from DETR, whish has following modification compared to original transformer (torch.nn.Transformer):
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
""" 

import copy
from typing import Optional, List
from jsonargparse import ArgumentParser
# from siamcar.models.position_encoding import PositionEmbeddingSine
import torch
import torch.nn.functional as F
from torch import nn, Tensor

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from colorama import Fore, Style
from typing import Optional, List
from torch import Tensor 

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.segment_embdded_factor = 0.0  #0.5, 0.0 gives best result for VOT2018. TODO: hyperparameter

    def forward(self, tensor_list: NestedTensor, multi_frame = False):
        x = tensor_list

        #print("x: {}".format(x.shape))
        
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        #  my change 
        b,c,h,w=tensor_list.shape 
        not_mask = torch.ones([b,h,w], device = tensor_list.device) # Adding position embedding to maks area (average padding area) will degrade the performance
        
        # Note: can not use different model between training and inference
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        #print("x_embed: {}".format(x_embed.shape))
        #print("x_embed[:, :, -1:]: {}".format(x_embed[:, :, -1:].shape))
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        #print("x_embed: {}".format(x_embed.shape))
        #print("y_embed: {}".format(y_embed.shape))
        #print("dim_t: {}".format(dim_t))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        #print("pos_x: {}".format(pos_x.shape))
        #print("pos_y: {}".format(pos_y.shape))


        #print("stack: {}".format(torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).shape))
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        #print("pos_x: {}".format(pos_x.shape))
        #print("pos_y: {}".format(pos_y.shape))

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        #print("pos: {}".format(pos.shape))

        # add an additioanl segment (frame) embedding for multilple frames.
        if multi_frame:
            # a temporal segment embedding for two frames => TODO: should learn?
            assert(len(pos) == 2)
            pos[-1].add_(self.segment_embdded_factor)

        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

def build_position_encoding(args):
    N_steps = args.transformer.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding

class Transformer(nn.Module): 

    def __init__(self, cfg, d_model=256, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=512, dropout=0.0,
                 activation="relu", normalize_before=False, 
                 return_intermediate_dec=False):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        self.pos_encoder=PositionEmbeddingSine()

        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
        self.memory = [] 

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # [1, 256, 16, 16], None,  [1, 256, 16, 16], [1, 256,32,32],None, [1, 256, 32,32]
    def forward(self, template_src, search_src):
        # get pos  encoding 
        template_mask,search_mask=None,None # 

        template_pos_embed = self.pos_encoder(template_src) #[64, 96, 4, 4] --> []
        search_pos_embed = self.pos_encoder(search_src)
    
        #def forward(self, template_src, template_mask, template_pos_embed, search_src, search_mask, search_pos_embed, memory = None):
        
        # flatten and permute bNxCxHxW to HWxbNxC for encoder in transformer
        template_src = template_src.flatten(2).permute(2, 0, 1) #[1, 96, 4, 4] --> [16, 16, 96]
        template_pos_embed = template_pos_embed.flatten(2).permute(2, 0, 1) #[1, 96, 4, 4] --> [16, 96, 96]
        if template_mask is not None:
            template_mask = template_mask.flatten(1)
        
        # encoding the template embedding with positional embbeding
        # if self.memory is None:
        memory = self.encoder(template_src, src_key_padding_mask=template_mask, pos=template_pos_embed) 

        # [256, 1, 256] 
        # flatten and permute bNxCxHxW to HWxbNxC for decoder in transformer
        search_src = search_src.flatten(2).permute(2, 0, 1) # tgt  [1, 256, 32, 32] --> [1024, 1, 256]
        search_pos_embed = search_pos_embed.flatten(2).permute(2, 0, 1)# [1, 256, 32, 32] --> [1024, 1, 256]
        if template_mask is not None:
            search_mask = search_mask.flatten(1)
        # [1024, 1, 256], [256, 1, 256], None, None, [256, 1, 256],[ 1024, 1, 256] --> [1, 1024, 1, 256]
        hs = self.decoder(search_src, memory,
                          memory_key_padding_mask=template_mask,
                          tgt_key_padding_mask=search_mask,
                          encoder_pos=template_pos_embed,
                          decoder_pos=search_pos_embed)          
        
        return  hs.transpose(1, 2)  # [1, 256, 16, 96]
        #return  hs.transpose(1,2).(2,3).squeeze(0)  #[1, 256, 16, 96]
        
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                encoder_pos: Optional[Tensor] = None,
                decoder_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           encoder_pos=encoder_pos,
                           decoder_pos=decoder_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False): #
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weight_map = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                               key_padding_mask=src_key_padding_mask)
        #print("encoder: self attn_weight_map: {}".format(attn_weight_map))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     encoder_pos: Optional[Tensor] = None,
                     decoder_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, decoder_pos)
        tgt2, attn_weight_map = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        #print("decoder: self attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_weight_map = self.multihead_attn(query=self.with_pos_embed(tgt, decoder_pos),
                                                          key=self.with_pos_embed(memory, encoder_pos),
                                                          value=memory, attn_mask=memory_mask,
                                                          key_padding_mask=memory_key_padding_mask)
        #print("decoder: multihead attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    encoder_pos: Optional[Tensor] = None,
                    decoder_pos: Optional[Tensor] = None): 
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, decoder_pos)
        tgt2, attn_weight_map = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        #print("decoder: self attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn_weight_map = self.multihead_attn(query=self.with_pos_embed(tgt2, decoder_pos),
                                                    key=self.with_pos_embed(memory, encoder_pos),
                                                    value=memory, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask)
        #print("decoder: multihead attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                encoder_pos: Optional[Tensor] = None,
                decoder_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    encoder_pos, decoder_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 encoder_pos, decoder_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_args_parser():
    parser = ArgumentParser(prog='transformer')

    parser.add_argument('--enc_layers', type=int, default=1,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', type=int, default=1,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--nheads', type=int, default=8,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout applied in the transformer")
    parser.add_argument('--pre_norm', type=bool, default=False,
                        help="whether do layer normzalize before attention mechansim")

    return parser


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

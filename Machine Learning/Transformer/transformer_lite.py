"""
Implementation of the simple Transformer model
Copyright 2025. Peizhi Yan
"""
import torch
import torch.nn as nn
import math
import numpy as np

from transformer_utils import PositionalEncoding, generate_mask


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer"""
    def __init__(self, dim=256, num_heads=8):
        """
            - dim: the dimension of the input tensor [N, L, D]
            - num_heads: the number of (parallel) attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads               # the number of attention heads (H)
        self.dim = dim                           # the model dimension
        self.dim_k = self.dim // self.num_heads  # the dimension of key and query tensors
        self.dim_v = self.dim_k                  # the dimension of value tensor
        self.Wq = nn.Linear(dim, num_heads * self.dim_k, bias=False)  # [D, H*Dk]
        self.Wk = nn.Linear(dim, num_heads * self.dim_k, bias=False)  # [D, H*Dk]
        self.Wv = nn.Linear(dim, num_heads * self.dim_v, bias=False)  # [D, H*Dv]
        self.out = nn.Linear(num_heads * self.dim_v, dim, bias=False) # output linear layer weights  [H*Dv, D]

    def attention(self, Q, K, V, mask_2d=None):
        """Scaled Dot-Product Attention
        Inputs:
            - Q: query tensor [N, L, D]
            - K: key tensor   [N, L, D]
            - V: value tensor [N, L, D]
            - mask_2d: optional tensor [N, L, L] or broadcastable to [N, L, L]
        Outputs:
            - output tensor [N, L, D]
            - attention weights [N, L, L]
        """
        KT = K.transpose(-2, -1)             # transpose of K     [N, D, L]
        QK = Q @ KT / math.sqrt(Q.shape[-1]) # QK / sqrt(D)       [N, L, L]
        if mask_2d is not None:
            # mask out positions by setting them to a very large negative value
            #QK = QK.masked_fill(mask_2d == 0, float('-inf'))
            QK = QK.masked_fill(mask_2d == 0, float(0))
        A = torch.softmax(QK, dim=-1)        # attention weights  [N, L, L]
        return A @ V, A

    def forward(self, Xq, Xk, Xv, mask_2d=None, return_attention=False):
        """
        Inputs:
            - Xq: [N, L, D] the input tensor for computing Q
            - Xk: [N, L, D] the input tensor for computing K
            - Xv: [N, L, D] the input tensor for computing V
            - mask_2d: optional tensor [N, L, L] or broadcastable to [N, L, L]
            - return_attention: whether to return attention weights
        Outputs:
            - output tensor [N, L, D]
            - (optional) attention tensor [N, L, H, Dv]
        """
        # compute Q, K, and V
        Q = self.Wq(Xq)  # Q = Xq * Wq   [N, L, H*Dk]
        K = self.Wk(Xk)  # K = Xk * Wk   [N, L, H*Dk]
        V = self.Wv(Xv)  # V = Xv * Wv   [N, L, H*Dv]
        
        # multi head attention
        attn_weights = []
        attn_outputs = []
        for h in range(self.num_heads):
            Qh = Q[:,:,h*self.dim_k:(h+1)*self.dim_k]     # [N, L, Dk]
            Kh = K[:,:,h*self.dim_k:(h+1)*self.dim_k]     # [N, L, Dk]
            Vh = V[:,:,h*self.dim_v:(h+1)*self.dim_v]     # [N, L, Dv]
            Oh, Ah = self.attention(Qh, Kh, Vh, mask_2d)  # [N, L, Dv], [N, L, L] 
            attn_weights.append(Ah)
            attn_outputs.append(Oh)

        # concatenation
        attn_output = torch.cat(attn_outputs, dim=-1)  # [N, L, D]  (D=H*Dv)

        # linear output layer
        output = self.out(attn_output)                 # [N, L, D]

        if return_attention:
            return output, attn_weights
        else:
            return output


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Neural Network"""
    def __init__(self, dim=256, hidden_dim=512):
        """
        Inputs:
            - dim: the dimension of the input tensor [N, L, D]
            - hidden_dim: the dimension of the hidden layer
        """
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, X):
        """
        Inputs:
            - X: the input tensor [N, L, D]
        Outputs:
            - the output tensor   [N, L, D]
        """
        return self.fc2(torch.relu(self.fc1(X)))


class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim=256, num_heads=8, hidden_dim=512):
        """
        - dim: the embedded token dimension
        - num_heads: the number of attention heads
        - hidden_dim: the hidden layer dimension for feed-forword network
        """
        super(EncoderBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, num_heads) # multi-head attention block
        self.ffn = FeedForwardNetwork(dim, hidden_dim) # feed-forward network
        self.ln1 = nn.LayerNorm(normalized_shape=[dim], eps=1e-6) # layer normalization on the last dimension
        self.ln2 = nn.LayerNorm(normalized_shape=[dim], eps=1e-6) # layer normalization on the last dimension

    def forward(self, X):
        """
        Inputs:
            - X: the input tensor [N, L, D]
        Outputs:
            - output: the output tensor   [N, L, D]
        """
        # self-attention
        Z = self.attn(Xq=X, Xk=X, Xv=X) # [N, L, D]

        # add & layer norm
        Z = Z + X                       # [N, L, D]
        Z = self.ln1(Z)

        # feed-forward network
        output = self.ffn(Z)

        # add & layer norm
        output = output + Z
        output = self.ln2(output)

        return output


class DecoderBlock(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, dim=256, num_heads=8, hidden_dim=512):
        """
        - dim: the embedded token dimension
        - num_heads: the number of attention heads
        - hidden_dim: the hidden layer dimension for feed-forword network
        """
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)  # multi-head attention block for self attention
        self.cross_attn = MultiHeadAttention(dim, num_heads) # multi-head attention block for cross attention
        self.ffn = FeedForwardNetwork(dim, hidden_dim) # feed-forward network
        self.ln1 = nn.LayerNorm(normalized_shape=[dim], eps=1e-6) # layer normalization on the last dimension
        self.ln2 = nn.LayerNorm(normalized_shape=[dim], eps=1e-6) # layer normalization on the last dimension
        self.ln3 = nn.LayerNorm(normalized_shape=[dim], eps=1e-6) # layer normalization on the last dimension

    def forward(self, Y, Z, mask_2d):
        """
        Inputs:
            - Y: decoder's previous output [N, L, D]
            - Z: encoder's output          [N, L, D]
            - mask_2d: [N, L, L]  uint8  (0,1)
        Outputs:
            - output: the output tensor    [N, L, D]
        """
        # masked self-attention
        Y_attn = self.self_attn(Xq=Y, Xk=Y, Xv=Y, mask_2d=mask_2d) # [N, L, D]

        # add & layer norm
        Y_attn = Y_attn + Y                # [N, L, D]
        Y_attn = self.ln1(Y_attn)

        # cross-attention (key and value from encoder, query from decoder)
        Z_attn = self.cross_attn(Xq=Y_attn, Xk=Z, Xv=Z)  # [N, L, D]

        # add & layer norm
        Z_attn = Z_attn + Y_attn           # [N, L, D]
        Z_attn = self.ln2(Z_attn)

        # feed-forward network
        output = self.ffn(Z_attn)

        # add & layer norm
        output = output + Z_attn
        output = self.ln3(output)

        return output


class Transformer(nn.Module):
    """Transformer Network"""
    def __init__(self, dim=256, num_heads=8, num_blocks=6, hidden_dim=512, 
                       max_length=1000, vocabulary_size=5000):
        """
        - dim: the embedded token dimension
        - num_heads: the number of attention heads
        - num_blocks: the number of encoder/decoder blocks
        - hidden_dim: the hidden layer dimension for feed-forword network
        - max_length: the maximum sequence length (number of tokens)
        """
        super(Transformer, self).__init__()
        # register hyperparameters
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        
        # positional encoding module
        self.positional_encoding = PositionalEncoding(max_length, dim)

        # transformer encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(dim=dim, num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_blocks)]
        )

        # transformer decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(dim=dim, num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_blocks)]
        )

        # output layer for token generation
        self.output_layer = nn.Linear(dim, self.vocabulary_size, bias=True)

    def encode(self, X):
        """
        Inputs:
            - X: input sequences of embedded tokens [N, L, D]
        Outputs:
            - Z: output sequences of encoded features
        """
        # apply positional encoding
        Z = X + self.positional_encoding(X)

        # pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            Z = encoder_block(Z)
        
        return Z

    def decode(self, Y, Z, mask):
        """
        Inputs:
            - Y: decoder's previous output [N, L, D]
            - Z: encoder's output          [N, L, D]
            - mask: [N, L]  uint8  (0,1)  the input token mask
        Outputs:
            - output: output sequences of decoded features
        """
        # generate attention masks
        mask_2d = generate_mask(mask)

        # apply positional encoding
        output = Y + self.positional_encoding(Y)

        # pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            output = decoder_block(Y=output, Z=Z, mask_2d=mask_2d)

        return output


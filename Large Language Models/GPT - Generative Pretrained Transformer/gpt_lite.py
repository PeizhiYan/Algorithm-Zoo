"""
Implementation of the simple GPT model
Copyright 2025. Peizhi Yan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from gpt_utils import PositionalEncoding


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
        KT = K.transpose(-2, -1)                # transpose of K     [N, D, L]
        # QK = Q @ KT / math.sqrt(Q.shape[-1])  # QK / sqrt(D)       [N, L, L]
        QK = Q @ KT * Q.shape[-1]**-0.5         # QK / sqrt(D)       [N, L, L]
        if mask_2d is not None:
            # mask out positions by setting them to a very large negative value
            QK = QK.masked_fill(mask_2d == 0, float('-inf'))       # [N, L, L]
        A = torch.softmax(QK, dim=-1)           # attention weights  [N, L, L]
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
            Qh = Q[:,:,h*self.dim_k:(h+1)*self.dim_k]     # [N, L, D]
            Kh = K[:,:,h*self.dim_k:(h+1)*self.dim_k]     # [N, L, Dk]
            Vh = V[:,:,h*self.dim_v:(h+1)*self.dim_v]     # [N, L, Dv]
            Oh, Ah = self.attention(Qh, Kh, Vh, mask_2d)  # [N, L, Dv], [N, L, L]
            attn_weights.append(Ah)
            attn_outputs.append(Oh)

        # concatenation
        attn_output = torch.cat(attn_outputs, dim=-1)     # [N, L, D]  (D=H*Dv)

        # linear output layer
        output = self.out(attn_output)                    # [N, L, D]

        if return_attention:
            return output, attn_weights
        else:
            return output


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Neural Network / MLP"""
    def __init__(self, dim=256, hidden_dim=512):
        """
        Inputs:
            - dim: the dimension of the input tensor [N, L, D]
            - hidden_dim: the dimension of the hidden layer
        """
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        """
        Inputs:
            - x: the input tensor [N, L, D]
        Outputs:
            - the output tensor   [N, L, D]
        """
        return self.fc2(torch.relu(self.fc1(x)))


class GPTBlock(nn.Module):
    """GPT Decoder Block"""
    def __init__(self, dim=256, num_heads=8):
        """
        - dim: the embedded token dimension (D)
        - num_heads: the number of attention heads
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)  # multi-head attention block for self attention
        self.ln1 = nn.LayerNorm(normalized_shape=[dim], eps=1e-6) # layer normalization on the last dimension
        self.ln2 = nn.LayerNorm(normalized_shape=[dim], eps=1e-6) # layer normalization on the last dimension
        self.ffn = FeedForwardNetwork(dim, dim) # feed-forward network

    def forward(self, x, mask_2d):
        """
        Inputs:
            - x: input to GPT's decoder [N, L, D]
            - mask_2d: [N, L, L]  uint8  (0,1)
        Outputs:
            - output: the output tensor [N, L, D]
        """
        # layer norm
        x_ = self.ln1(x)                   # [N, L, D]
        # masked self-attention & add
        x_attn = x + self.self_attn(Xq=x_, Xk=x_, Xv=x_, mask_2d=mask_2d) # [N, L, D]
        # layer norm
        x_attn_ = self.ln2(x_attn)         # [N, L, D]
        # mlp & add
        output = x_attn + self.ffn(x_attn_) # [N, L, D]
        return output


class GPT(nn.Module):
    """GPT Network"""
    def __init__(self, dim=256, num_heads=8, num_blocks=6, 
                       max_length=1000, vocabulary_size=5000, padding_idx=-1, 
                       device='cpu'):
        """
        - dim: the embedded token dimension (D)
        - num_heads: the number of attention heads
        - num_blocks: the number of encoder/decoder blocks
        - max_length: the maximum sequence length (L_max)
        - vocabulary_size: the size of the vocabulary (V)
        - padding_idx: the index of the padding token, in our case, the <EMPTY> token
        """
        super().__init__()
        # register hyperparameters
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        self.padding_idx = padding_idx

        # register the lower-triangle mask
        self.register_buffer('tril', torch.tril(torch.ones(max_length, max_length)))
        
        # embedding layer for input tokens
        self.token_embeder = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=dim, 
                                                padding_idx=padding_idx, device=device)
        # positional encoding module
        self.positional_encoding = PositionalEncoding(max_length, dim, device)
        # transformer decoder blocks
        self.gpt_blocks = nn.ModuleList(
            [GPTBlock(dim=dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        # output layer for token generation
        self.output_layer = nn.Linear(dim, self.vocabulary_size, bias=True)
        # initialize weights
        self.apply(self._init_weights)
        # register device
        self.device = device
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def decode(self, x):
        """
        Inputs:
            # - x: decoder's previous output [N, L, D]  (L <= L_max)
            - x: input tensor [N, L]  torch.long  (0,1)  the token indices
            # - mask_2d: [N, L, L]  uint8  (0,1)  the lower-triangle casual mask
        Outputs:
            - logits: output sequences of decoded logits [N, L, V]
        """
        N, L = x.shape
        # embedding layer
        x = self.token_embeder(x)           # [N, L, D]
        # apply positional encoding
        x = x + self.positional_encoding(x)
        # pass through the decoder blocks
        for gpt_block in self.gpt_blocks:
            x = gpt_block(x=x, mask_2d=self.tril[:L, :L])
        # output layer
        logits = self.output_layer(x)      # [N, L, V]
        return logits
    
    @torch.no_grad()
    def generate(self, tokenizer, context=None, generate_length=100, narrate=True):
        """
        Generate text using the GPT model
        Inputs:
            - tokenizer: the tokenizer object
            - context: the input sequence [N, T]  torch.long  (0,1)  the token indices, we just assume N=1
            - generate_length: the number of tokens to generate         
            - narrate: whether to print the generated tokens
        Outputs:
            - generated_tokens: the generated tokens [N, T]  torch.long  (0,1)  the token indices
        """
        if generate_length > self.max_length:
            print(f"Warning: the maximum length is {self.max_length}.")
            generate_length = self.max_length
        if context is None:
            context = torch.ones((1, 1), dtype=torch.long, device=self.device)
        for _ in range(generate_length):
            # crop idx to the last block_size tokens
            idx_cond = context[:, -self.max_length:] # [N, T]
            # get the predictions
            logits = self.decode(idx_cond)           # [N, L, V] (V = vocabulary size)
            # focus only on the last time step
            logits = logits[:, -1, :]                # [N, V]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)        # [N, V]
            # sample from the distribution (this ensures diversity of generation)
            idx_next = torch.multinomial(probs, num_samples=1)  # [N, 1]
            # append sampled index to the running sequence
            context = torch.cat((context, idx_next), dim=1)     # [N, T+1]
            generated_token_indices = context[0, :].cpu().numpy()      # [T+1] list object
            if narrate:
                print(f"\r", "".join(tokenizer.get_tokens(generated_token_indices)), end='', flush=True)
        return generated_token_indices


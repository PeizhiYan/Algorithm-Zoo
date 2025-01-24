"""
Utility functions for Transformer model
Copyright 2025. Peizhi Yan
"""

import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, max_length=5000, dim=256):
        """
        Inputs:
            - max_length: the maximum length of the input tensor [N, L, D], (L <= max_length)
            - dim: the dimension D of the input tensor [N, L, D]
        Formula:
            PE(pos, 2i)   = sin(pos / 10000^(2i / D))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / D))
        """
        super(PositionalEncoding, self).__init__()
        self.max_length = max_length
        self.dim = dim
        PE = torch.zeros(max_length, dim) # [max_length, D]
        positions = torch.arange(start=0, end=max_length, step=1, dtype=torch.float32).unsqueeze(1)    # [max_length, 1]
        dividors = torch.pow(10000, torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim) # [D//2]
        #dividors = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # this is mathematically equivalent
        PE[:, 0::2] = torch.sin(positions / dividors) # for even dimensions
        PE[:, 1::2] = torch.cos(positions / dividors) # for odd dimensions
        self.PE = PE.detach()

    def forward(self, x):
        """
        Add positional encoding to the input tensor
        ------------------------------------------
        Inputs:
            - x: the input tensor [N, L, D]
        """
        assert x.shape[1] <= self.max_length
        N, L, D = x.shape
        return x + self.PE[:L, :].unsqueeze(0).expand(N, L, D)


@torch.no_grad()
def generate_mask(mask):
    """Generate the mask used in next token generation tasks
    Inputs:
        - mask: [N, L]  uint8  (0,1)  the input token mask
    Outputs:
        - the output mask tensor [N, L, L]  uint8 (0,1)
    """
    batch_size = mask.shape[0]
    seq_length = mask.shape[1]
    mask_2d = torch.ones([batch_size, seq_length, seq_length], dtype=torch.float32).to(mask.device)
    for b in range(batch_size):
        zero_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]  # Indices where mask[b] == 0
        mask_2d[b] = torch.tril(mask_2d[b], diagonal=0)
        mask_2d[b, zero_indices, :] = 0  # mask out rows
        mask_2d[b, :, zero_indices] = 0  # mask out columns
    return mask_2d

class SimpleTokenizer:
    """
    Each character will be converted to a token
    """
    def __init__(self, texts : list, max_length : int = 1000):
        # - texts: list of strings in the entire corpus
        # - max_length: maximum number of tokens in each sequence
        self.max_length = max_length
        self.token_to_index = {}
        self.index_to_token = []
        self.special_tokens = ['<EMPTY>', '<BEGIN>', '<EOS>', '<UNKNOWN>']
        for token in self.special_tokens:
            self.token_to_index[token] = len(self.index_to_token)
            self.index_to_token.append(token)
        for s in texts:
            tokens = self.tokenize(text=s, init=True)
            for token in tokens:
                if token not in self.index_to_token:
                    self.token_to_index[token] = len(self.index_to_token)
                    self.index_to_token.append(token)
        self.num_tokens = len(self.index_to_token) # the vocabulary size

    def tokenize(self, text : str, init : bool = False):
        # convert each character to a token
        tokens = []
        if init:
            for c in text: tokens.append(c)
        else:
            tokens.append('<BEGIN>') # begin of a sequence
            valid_tokens = self.tokenize(text=text, init=True)
            tokens = tokens + valid_tokens
            tokens.append('<EOS>') # end of a sequence
            while len(tokens) < self.max_length:
                tokens.append('<EMPTY>') # add empty padding to extend to maximum length
            tokens = tokens[:self.max_length] # trim-off the exceeding part
        return tokens

    def get_indices(self, text : str):
        # give a text string, get the indices of tokens
        indices = []
        for token in self.tokenize(text=text):
            indices.append(self.token_to_index[token])
        return indices
    
    def get_tokens(self, indices : list):
        # given a list of indices, return the list of tokens
        tokens = []
        for idx in indices:
            if idx >=0 and idx < self.num_tokens:
                tokens.append(self.index_to_token[idx])
            else:
                tokens.append('<UNKNOWN>') # unknown/invalid token
        return tokens

    def get_text_from_tokens(self, tokens : list):
        # given a list of tokens, return the text
        text = ''
        for token in tokens:
            if token == '<BEGIN>':
                pass
            elif token in ['<EMPTY>', '<UNKNOWN>']:
                pass
            elif token == '<EOS>':
                break
            else:
                text += token
        return text

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from transformer_utils import PositionalEncoding, generate_mask
from transformer_lite import EncoderBlock, DecoderBlock, Transformer

device = 'cpu'

# define some hyperparameters
BATCH_SIZE = 16
MAX_LENGTH = 1000   # the maximum amount of "tokens" in a sequence
DIMENSION = 128     # the dimension of each "token"
HIDDEN_DIM = 512    # the hidden dimension of the FFN
ATTN_HEADS = 8      # the number of attention heads


def main():

    # random fake input
    x = torch.randn([BATCH_SIZE, 100, DIMENSION]).to(device)
    y = torch.randn([BATCH_SIZE, 200, DIMENSION]).to(device)
    mask_2d = generate_mask(mask=torch.ones([BATCH_SIZE, 200], dtype=torch.uint8))
    mask = torch.ones([BATCH_SIZE, 200], dtype=torch.uint8)
    # x = torch.zeros([BATCH_SIZE, 200, DIMENSION]).to(device)

    #print(np.sin(0))

    max_length = MAX_LENGTH
    dim = DIMENSION
    PE = PositionalEncoding(max_length, dim).to(device)
    xe = PE(x)

    plt.imshow(xe[0])
    plt.show()

    # mask_ = torch.tensor([[1,1,1,1,1,1], [0,0,1,1,0,1]])
    # mask = generate_mask(mask_)
    # plt.imshow(mask[0])
    # plt.show()
    # plt.imshow(mask[1])
    # plt.show()

    # encoder = EncoderBlock(dim=DIMENSION, num_heads=ATTN_HEADS, hidden_dim=HIDDEN_DIM)
    # decoder = DecoderBlock(dim=DIMENSION, num_heads=ATTN_HEADS, hidden_dim=HIDDEN_DIM)
    # z = encoder(X=xe)
    # output = decoder(Y=y, Z=z, mask_2d=mask_2d)
    # with torch.no_grad():
    #     plt.imshow(output[0])
    #     plt.show()


    transformer = Transformer(dim=DIMENSION, num_heads=ATTN_HEADS, num_blocks=6, 
                              hidden_dim=HIDDEN_DIM, max_length=MAX_LENGTH)

    Z = transformer.encode(X=x)
    Z = transformer.decode(Y=y, Z=Z, mask=mask)
    with torch.no_grad():
        plt.imshow(Z[0])
        plt.show()


if __name__ == '__main__':
    main()
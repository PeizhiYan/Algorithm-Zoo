################################
# Author: Peizhi Yan           #
# Last Modified on: 11/17/2020 #
################################
"""
Important Notice: 
    Currently this method only supports .bmp grayscale image!
"""

import numpy as np
from pca import PCA
import pickle
import cv2

def load_bmp(path):
    """
    Load a .bmp grayscale image
        path - the path of bitmap image
    """
    img = cv2.imread(path, 0)/255.
    return img

def compress(img, save_path, K, PRECISION = 1e5):
    """
    Compress the image and save the processed file
              img - the loaded image
        save_path - the path to save the compressed file 
                K - the number of principle components to keep
        PRECISION - for converting data to integer
    """
    H = img.shape[0] # image height
    W = img.shape[1] # image width
    pca = PCA()
    print('>>> compressing')
    cmp = pca.fit(x = img, k = K) # compressed image
    print('>>> saving')
    buffer = {
        'cmp': (cmp*PRECISION).astype(int),
        'pc': (pca.pc[:,:K]*PRECISION).astype(int),
        'mu': pca.mu,
        'W': W, # original image width
        'H': H, # original image height
        'K': K,  # compressed dimension
        'P': PRECISION
    }
    with open(save_path, 'wb') as handle:
        pickle.dump(buffer, handle)
    print('>>> done!')

def reconstruct(load_path):
    """
    Reconstruct the image from the compressed file
    """
    print(">>> loading")
    # Load compressed image
    with open(load_path, 'rb') as handle:
        loaded_buffer = pickle.load(handle)
    PRECISION = loaded_buffer['P']
    cmp = loaded_buffer['cmp'].astype(float)/PRECISION
    pc = loaded_buffer['pc'].astype(float)/PRECISION
    mu = loaded_buffer['mu']
    W = loaded_buffer['W']
    H = loaded_buffer['H']
    K = loaded_buffer['K']
    print(">>> reconstructing")
    # Reconstruct image
    img_reconstruct = np.zeros([H,W], dtype=int)
    pca = PCA()
    pca.pc = np.zeros([W,W])
    pca.pc[:,:K] = pc[:,:K]
    pca.mu = mu
    reconstruct = pca.inverse_transform(cmp)
    reconstruct = np.clip(reconstruct, 0, 1.0)
    img_reconstruct[:,:] = (reconstruct[:,:]*255).astype(int)
    print(">>> done")
    return img_reconstruct

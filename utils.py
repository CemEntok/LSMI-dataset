import torch
import numpy as np
import matplotlib.pyplot as plt

def apply_wb(org_img,pred,pred_type):
    pred_rgb = torch.zeros_like(org_img) # b,c,h,w

    if pred_type == "illumination":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,0,:,:] * (1 / (pred[:,0,:,:]+1e-8))    # R_wb = R * (1/illum_R)
        pred_rgb[:,2,:,:] = org_img[:,2,:,:] * (1 / (pred[:,2,:,:]+1e-8))    # B_wb = B * (1/illum_B)
    elif pred_type == "uv":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
        pred_rgb[:,2,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)
    
    return pred_rgb

def rgb2uvl(img_rgb):
        epsilon = 1e-8
        img_uvl = np.zeros_like(img_rgb, dtype='float32')
        img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
        img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
        img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

        return img_uvl

def plot_illum(pred_map=None,gt_map=None):
    fig = plt.figure()
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'ro')
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'bx')

    minx,miny = min(gt_map[:,0]),min(gt_map[:,1])
    maxx,maxy = max(gt_map[:,0]),max(gt_map[:,1])
    lenx = (maxx-minx)/2
    leny = (maxy-miny)/2
    add_len = max(lenx,leny) + 0.3

    center_x = (maxx+minx)/2
    center_y = (maxy+miny)/2

    plt.xlim(center_x-add_len,center_x+add_len)
    plt.ylim(center_y-add_len,center_y+add_len)

    # make square
    plt.gca().set_aspect('equal', adjustable='box')

    plt.close()

    fig.canvas.draw()

    return np.array(fig.canvas.renderer._renderer)

def mix_chroma(mixmap,chroma_list,illum_count):
    # ret = np.stack((np.zeros_like(mixmap[:,:,0],dtype=np.float),)*3, axis=2)
    ret = np.stack((np.zeros_like(mixmap[:,:,0],dtype=np.float64),)*3, axis=2)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i])-1
        mixmap_3ch = np.stack((mixmap[:,:,i],)*3, axis=2)
        ret += (mixmap_3ch * [[chroma_list[illum_idx]]])
    
    return ret


def insertNoise(alpha):
    import random
    import time  # Import the time module to get a different seed for each run

    # Set a random seed based on the current time
    random.seed(int(time.time()))

    # Define your parameter and noise characteristics
    # alpha = 0.5  # Example parameter value within [0, 1]
    std_dev = alpha / 10  # Adjust the standard deviation to control noise strength

    # Generate noise from a centered Gaussian distribution (mean=0)
    noise = random.gauss(0, std_dev)

    # Add the noise to the parameter
    noisy_parameter = alpha + noise

    # Ensure that the noisy parameter remains within [0, 1]
    noisy_parameter = np.maximum(0, np.minimum(1, noisy_parameter))

    # Create the complementary 2x2x1 matrix
    alphaComplement = 1 - noisy_parameter

    # Stack the two matrices along a new axis (3rd dimension)
    stacked_matrices = np.stack([noisy_parameter, alphaComplement], axis=2)

    # Ensure the sum of the two matrices is equal to 1
    assert np.allclose(np.sum(stacked_matrices, axis=2), 1.0)
    # print("noisy_parameter:")
    # print(noisy_parameter)
    # print("1-alpha:")
    # print(alphaComplement)
    # print("Stacked Matrices:")
    # print(stacked_matrices)
    # print(noisy_parameter)
    return stacked_matrices
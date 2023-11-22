import torch,rawpy
import numpy as np
import matplotlib.pyplot as plt

def apply_wb(org_img,pred,pred_type):
    """
    By using pred tensor (illumination map or uv),
    apply wb into original image (3-channel RGB image).
    """
    pred_rgb = torch.zeros_like(org_img) # b,c,h,w

    if pred_type == "illumination":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,0,:,:] / (pred[:,0,:,:]+1e-8)    # R_wb = R / illum_R
        pred_rgb[:,2,:,:] = org_img[:,2,:,:] / (pred[:,2,:,:]+1e-8)    # B_wb = B / illum_B
    elif pred_type == "uv":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
        pred_rgb[:,2,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)
    
    return pred_rgb

def rgb2uvl(img_rgb):
    """
    convert 3 channel rgb image into uvl
    """
    epsilon = 1e-8
    img_uvl = np.zeros_like(img_rgb, dtype='float32')
    img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
    img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
    img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

    return img_uvl

# def plot_illum(pred_map=None,gt_map=None,MAE_illum=None,MAE_rgb=None,PSNR=None):
#     """
#     plot illumination map into R,G 2-D space
#     """

#     fig = plt.figure()
#     if pred_map is not None:
#         plt.plot(pred_map[:,0],pred_map[:,1],'ro')
#     if gt_map is not None:
#         plt.plot(gt_map[:,0],gt_map[:,1],'bx')
#     plt.xlim(0,3)
#     plt.ylim(0,3)
#     plt.title(f'MAE_illum:{MAE_illum:.4f} / PSNR:{PSNR:.4f}')
#     plt.close()

#     fig.canvas.draw()

#     return np.array(fig.canvas.renderer._renderer)

def plot_illum(pred_map=None,gt_map=None,MAE_illum=None,MAE_rgb=None,PSNR=None):
    """
    plot illumination map into R,B 2-D space
    """

    # plot pred first, then gt
    fig = plt.figure()
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'bo',alpha=0.03,markersize=5)
        mean_pred = np.mean(pred_map, axis=0)
        plt.plot(mean_pred[0], mean_pred[1], 'go', markersize=8, label='Mean of pred_map')
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'ro',alpha=0.01,markersize=3)
    # breakpoint()
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
    plt.title(f'MAE_illum:{MAE_illum:.4f} / PSNR:{PSNR}')
    plt.close()

    fig.canvas.draw()
    plot_illum = np.array(fig.canvas.renderer._renderer)

    # plot gt first, then pred
    fig = plt.figure()
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'ro',alpha=0.01,markersize=3)
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'bo',alpha=0.03,markersize=5)
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
    plt.title(f'MAE_illum:{MAE_illum:.4f} / PSNR:{PSNR}')
    plt.close()

    fig.canvas.draw()
    plot_illum_rev = np.array(fig.canvas.renderer._renderer)
    # breakpoint()
    # from PIL import Image
    # import os
    # Image.fromarray(plot_illum).save(os.path.join('/home/cem/results/_test','_illum_map.png'))
    # Image.fromarray(plot_illum_rev).save(os.path.join('/home/cem/results/_test','_illum_map_rev.png'))
    return plot_illum, plot_illum_rev

# def countNeg(input):
#     """
#     Count negative value in input numpy.ndarray.
#     """
#     abc = [x for x in (input<0)]
#     negCount = np.sum(abc)
#     return negCount

def mix_chroma(mixmap,chroma_list,illum_count):
    """
    Mix illuminant chroma according to mixture map coefficient
    mixmap      : (w,h,c) - c is the number of valid illuminant
    chroma_list : (3 (RGB), 3 (Illum_idx))
                  contains R,G,B value or 0,0,0
    illum_count : contains valid illuminant number (1,2,3)
    """
    ret = np.stack((np.zeros_like(mixmap[:,:,0]),)*3, axis=2)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i])-1
        mixmap_3ch = np.stack((mixmap[:,:,i],)*3, axis=2)
        ret += (mixmap_3ch * [[chroma_list[illum_idx]]])
    
    return ret

def visualize(input_patch, pred_patch, gt_patch, templete, concat=True):
    """
    Visualize model inference result.
    1. Re-bayerize RGB image by duplicating G pixels.
    2. Copy bayer pattern image into rawpy templete instance
    3. Use user_wb to render RGB image
    4. Crop proper size of patch from rendered RGB image
    """
    # breakpoint()
    input_patch = input_patch.permute((1,2,0))
    pred_patch = pred_patch.permute((1,2,0))
    gt_patch = gt_patch.permute((1,2,0))

    height, width, _ = input_patch.shape
    # raw = rawpy.imread("../" + templete + ".dng")
    # breakpoint()
    raw = rawpy.imread("/home/cem/" + templete + ".dng")
    white_level = raw.white_level

    if templete == 'sony':
        black_level = 512
        white_level = raw.white_level / 4
    else:
        black_level = min(raw.black_level_per_channel)
        white_level = raw.white_level
        if templete == "jarno":
            # raw = rawpy.imread("/mnt/ssd-storage/cem/dataset/VisualTest/Place0_1.tiff")
            white_level = 65535.0
            black_level = 0
    # breakpoint()
    input_rgb = input_patch.numpy().astype('uint16')
    output_rgb = np.clip(pred_patch.cpu().numpy(), 0, white_level).astype('uint16')
    gt_rgb = gt_patch.numpy().astype('uint16') # original GT torch float32 tensor
    
    input_bayer = bayerize(input_rgb, templete, black_level)
    output_bayer = bayerize(output_rgb, templete, black_level)
    gt_bayer = bayerize(gt_rgb, templete, black_level)

    input_rendered = render(raw, white_level, input_bayer, height, width, "daylight_wb")
    output_rendered = render(raw, white_level, output_bayer, height, width, "maintain")
    gt_rendered = render(raw, white_level, gt_bayer, height, width, "maintain")
    # breakpoint()
    if concat:
        return np.hstack([input_rendered, output_rendered, gt_rendered])
    else:
        return input_rendered, output_rendered, gt_rendered, input_rgb, output_rgb, gt_rgb

def bayerize(img_rgb, camera, black_level):
    h,w,c = img_rgb.shape

    bayer_pattern = np.zeros((h*2,w*2))
    
    if camera == "galaxy":
        bayer_pattern[0::2,1::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,2] # B
    elif camera == "sony" or camera == 'nikon':
        bayer_pattern[0::2,0::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,2] # B

    return bayer_pattern + black_level

def render(raw, white_level, bayer, height, width, wb_method):
    raw_mat = raw.raw_image # 3000,4000 which is dng raw image size
    for h in range(height*2):
        for w in range(width*2):
            raw_mat[h,w] = bayer[h,w]

    if wb_method == "maintain": # except input, for output and GT
        user_wb = [1.,1.,1.,1.]
    elif wb_method == "daylight_wb": # for input only
        user_wb = raw.daylight_whitebalance # [1.9452837705612183, 1.0002996921539307, 1.5499752759933472, 0.0]

    rgb = raw.postprocess(user_sat=white_level, user_wb=user_wb, half_size=True, no_auto_bright=False)
    rgb_croped = rgb[0:height,0:width,:]
    # breakpoint()
    return rgb_croped

def countNeg(input):
    """
    Count negative value in input numpy.ndarray.
    """
    abc = [x for x in (input<0)]
    negCount = np.sum(abc)
    return negCount

def normEnder(img):
    from PIL import Image
    import numpy as np

    # Load the PNG image
    image = Image.open(img) # 'your_image.png'
    imgName = img.split(".")[0].split("/")[-1].split("_")[-2]
    # Convert sRGB values to linear RGB values
    linear_rgb_image = Image.new('RGB', image.size)
    linear_rgb_image.paste(image, (0, 0), image)

    # Map linear RGB values to the 0-255 range
    normalized_image = Image.eval(linear_rgb_image, lambda x: int(x * 255))

    # Optionally, apply gamma correction (gamma = 2.2)
    gamma_corrected_image = Image.eval(normalized_image, lambda x: int((x / 255.0) ** 2.2 * 255))

    # Save the final image
    gamma_corrected_image.save(imgName +'_RGB.png')

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

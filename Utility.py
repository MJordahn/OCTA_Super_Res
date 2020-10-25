import torch
import math
import numpy as np
import skimage


def calculate_scores(upsampled_ims, true_ims):
        diff_im = true_ims-upsampled_ims
        m = np.mean(diff_im.detach().numpy()**2)
        SSIM_real = true_ims.permute(1,2,3,0)[0,:,:,:].numpy()
        SSIM_upsampled = upsampled_ims.permute(1,2,3,0)[0,:,:,:].numpy()
        SSIM, SSSIM_IM = skimage.metrics.structural_similarity(SSIM_real, SSIM_upsampled,  gaussian_weights=True, data_range=1, multichannel=True, full=True, sigma=1, use_sample_covariance=False)
        PSNR = 10*math.log10(1**2/m)
        return PSNR, SSIM

# Saves generator & discriminator:
def saveModels(generator, discriminator, path="checkpoints/Pix2Pix", idx=-1):
    if idx == -1:
        torch.save(generator, path + "_generator")
        torch.save(discriminator, path + "_discriminator")
    else:
        torch.save(generator, path + "_generator_" + str(idx))
        torch.save(discriminator, path + "_generator_" + str(idx))
    return

# Loads generator & discriminator:
def readModels(generator_name, discriminator_name=None, gpu=True):
    cpu_device = torch.device('cpu')
    generator = torch.load(generator_name, map_location=cpu_device)
    print("Load: ", generator_name)
    if not(discriminator_name is None):
        discriminator = torch.load(discriminator_name, map_location=cpu_device)
        print("Load: ", discriminator_name)
    else:
        discriminator=None
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generator.to(device)
        if not(discriminator_name is None):
            discriminator.to(device)
        print("Models moved to GPU")
    else:
        print("Models kept on CPU")
    return generator, discriminator

import natsort, glob, pickle, torch
from collections import OrderedDict
import numpy as np
import os
from PIL import Image

import options.options as option
from models import create_model
from Measure import Measure, psnr
from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
from imresize import imresize
from skimage.metrics import structural_similarity as ssim


import Measure

def find_files(wildcard): return natsort.natsorted(glob.glob(wildcard, recursive=True))

from PIL import Image
#def imshow(array):
#display(Image.fromarray(array))

from test import load_model, fiFindByWildcard, imread

def pickleRead(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Convert to tensor
def t(array): 
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255

# convert to image
def rgb(t): 
    return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)


def validate_model():
    #'./confs/SRFlow_CelebA_8X.yml'
    conf_path = './confs/SRFlow_DF2K_4X.yml' #'./confs/SRFlow_DF2K_8X.yml'
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    model_path = '../experiments/train/models/73000_G.pth'
    model.load_network(load_path=model_path, network=model.netG)
    #model.to('cpu')
    #model, opt = load_model(conf_path)
    lq_paths = fiFindByWildcard(os.path.join(opt['dataroot_LR'], '*.png'))
    gt_paths = fiFindByWildcard(os.path.join(opt['dataroot_GT'], '*.png'))

    lqs = [imread(p) for p in lq_paths]
    gts = [imread(p) for p in gt_paths]

    

    measure = Measure.Measure()


    temperature = 0.8

    metrics = []


    for i in range(0,100):
        # Sample a super-resolution for a low-resolution image
        
        #ipdb.set_trace()
        print(i)
        lq = lqs[i]
        gt = gts[i]
        

        diff_x = int(lq.shape[0]%2)
        diff_y = int(lq.shape[1]%2)
        if diff_x == 1:
            lq = lq[:-diff_x]
            gt = gt[:-8*diff_x]
        if diff_y == 1:
            lq = lq[:,:-diff_y]
            gt = gt[:,:-8*diff_y]

        
        #lq_im = Image.fromarray(lq)


        
        #import ipdb
        #ipdb.set_trace()
        sr = rgb(model.get_sr(lq=t(lq), heat=temperature))

        
        #sr_im = Image.fromarray(sr)
        #imshow(sr)
        
        #psnr, ssim, lpips = measure.measure(sr, gt)
        psnr = 0
        lpips = 0
        #import ipdb
        #ipdb.set_trace()
        ssim_score,diff = ssim(sr,gt,full=True, multichannel=True,channel_axis=2,data_range=1)
        #np.save(f'metrics/iter_{i}.npy',[psnr,ssim,lpips])
        
        metrics.append([psnr,ssim_score,lpips])
    
    
    metrics = np.mean(np.array(metrics),axis=0)
    psnr, ssim_score, lpips = metrics[0],metrics[1],metrics[2]
    print(temperature, psnr, ssim_score, lpips)


def get_metrics():
    metrics = []
    for i in range(100):
        if os.path.exists(f'metrics/iter_{i}.npy'):
            metrics.append(np.load(f'metrics/iter_{i}.npy'))
    print(len(metrics))
    metrics = np.mean(np.array(metrics),axis=0)
    psnr, ssim, lpips = metrics[0],metrics[1],metrics[2]
    print('Temperature: {:0.2f} - PSNR: {:0.1f}, SSIM: {:0.1f}, LPIPS: {:0.2f}\n\n'.format(0.8, psnr, ssim, lpips))


if __name__ == '__main__':
    validate_model()
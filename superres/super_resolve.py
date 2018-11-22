from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import skimage
import skimage.measure
import math
import numpy as np
import cv2

def dump_intermediates(intermediates):
    for k, inter in enumerate(intermediates):
        dump_inter(k, inter)
    
def dump_inter(z, inter):
    inter = inter.cpu()
    inter_img_y = inter[0].detach().numpy()
    print('inter_img_y.shape = %s' % repr(inter_img_y.shape))
    inter_img_y *= 255.0
    inter_img_y = inter_img_y.clip(0, 255).astype(np.uint8)
    img_list = []
    for k in range(inter_img_y.shape[0]):
        img_list.append(inter_img_y[k])
    merged = cv2.merge(img_list)
    merged_bgr = cv2.cvtColor(merged, cv2.COLOR_RGBA2BGR)
    fname = "inter"+str(z)+".png"
    print('saving to %s' % fname)
    cv2.imwrite(fname, merged_bgr)
    #print('z = %d, %s' % (z, repr(img[8:16, 8:16])))



def super_resolve(input_image, model, output_filename, usecuda):
   img = Image.open(input_image).convert('YCbCr')
   y, cb, cr = img.split()

   img_to_tensor = ToTensor()
   input_ = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

   if usecuda:
       model = model.cuda()
       input_ = input_.cuda()

   out = model.forward(input_)
   out = out.cpu()
   out_img_y = out[0].detach().numpy()
   print('out_img_y.shape = %s' % repr(out_img_y.shape))
   out_img_y *= 255.0
   out_img_y = out_img_y.clip(0, 255)
   out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

   out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
   out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
   out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

   out_img.save(output_filename)
   print('output image saved to ', output_filename)
   return out_img.size

if __name__ == "__main__":
   # Training settings
   parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
   parser.add_argument('--input_image', type=str, required=True, help='input image to use')
   parser.add_argument('--model', type=str, required=True, help='model file to use')
   parser.add_argument('--output_filename', type=str, help='where to save the output image')
   parser.add_argument('--cuda', action='store_true', help='use cuda')
   opt = parser.parse_args()
   print(opt)
   
   model = torch.load(opt.model)
   super_resolve(opt.input_image, model, opt.output_filename, opt.cuda)

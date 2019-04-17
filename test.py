import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import torch.nn as nn
import os
from network import U_Net
from utils import search_file
import cv2
import numpy as np
from utils import  COLOR_DICT
from utils import dense_crf
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
num_channels = 3
batch_size = 4
size = (256, 256)
root = "data/membrane/test"
img_file = search_file(root, [".png"])
# print(img_file)
if __name__ == "__main__":
    model = U_Net(num_channels, num_classes).to(device)
    model.load_state_dict(torch.load('UNet_weights_bilinear_weight.pth'))
    model.eval()
    with torch.no_grad():
        for i in range(1):
            print(img_file[i])
            input = cv2.imread(img_file[i], cv2.IMREAD_COLOR)
            input = cv2.resize(input, size)
            original_img=input
            input = transforms.ToTensor()(input)
            input=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(input)
            input = input.view((-1,)+input.shape)
            # print(input.shape)
            output = model(input.to(device))
            # print(output[:,1].shape)
            # print(output.shape)
            result=np.zeros(size+(3,))
            new_mask=torch.argmax(output,dim=1)
            # new_mask=dense_crf(original_img,output[0,:,:,:].cpu().numpy())
            # print(new_mask.shape)
            # new_mask=(new_mask.cpu()).numpy()
            # print(new_mask.shape)
            # new_mask=new_mask.transpose(1,2,0)
            for i in range(256):
                for j in range(256):
                    # print(output[0][:,i,j])
                    # print(new_mask[i,j,:])
                    if new_mask[0,i,j]==0:
                        result[i,j]=COLOR_DICT[0]
                    else:
                        result[i, j] = COLOR_DICT[1]
            # result = cv2.resize(result, (512,512))
            cv2.namedWindow("test",cv2.WINDOW_NORMAL)
            cv2.imshow("test",result)
            cv2.imwrite("data/membrane/result/15.png",result)
            cv2.waitKey(0)
            # print(output[0])
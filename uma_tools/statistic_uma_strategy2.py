# --------------------------------------------------------
# The potential of cognitive-inspired neural network modeling framework for computer vision processing tasks
# Licensed under The MIT License [see LICENSE for details]
# Written by Guorun Li
# --------------------------------------------------------

# You can calculate the SSIM value of each image in your dataset by following this step: 
# 1. Using 'statistic_uma_strategy1.py' & 'statistic_uma_strategy2.py' to calculate the SSIM value of each image in the dataset.
#    Then you can get the Excel file (named ssim_origin_excel_file), which contains the SSIM value of each image in your dataset 
# 2. Using 'count_frequency.py' to calculate the frequency distribution P in your dataset.

import torch
import torch.fft
import numpy as np
import os
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class SSIMClass:
    def __init__(self, size):
        super().__init__()
        self.size = size
        for i in range(size):
            ssim_percentage = []
            setattr(self, f"ssim_percentage_{i+1}", ssim_percentage)
    def setvalue(self, dic):
        if len(dic) <= self.size and len(dic) > 0:
            ssim_list = getattr(self, f"ssim_percentage_{len(dic)}")
            ssim_list.append(dic)


def transfourier(tensor, filter_h, filter_w):
    x, y = tensor.shape
    # FFT
    dtf = torch.fft.rfft2(tensor)
    dtf_shift = torch.fft.fftshift(dtf)
    # Initialize the filter
    ffilter = torch.zeros((int(filter_h), int(filter_w))).cuda()
    target_height, target_width = x, y // 2 + 1
    # compute the padding
    pad_height = max(0, target_height - ffilter.shape[0])
    pad_width = max(0, target_width - ffilter.shape[1])
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    # padding
    fourier_filter = F.pad(ffilter, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=1)
    # filtering
    dtf_shift = dtf_shift * fourier_filter
    # IFFT
    f_ishift = torch.fft.ifftshift(dtf_shift)
    img_back = torch.fft.irfft2(f_ishift)
    # [vis, useful_paramaters]
    return 20 * torch.log(torch.abs(dtf_shift)), torch.abs(img_back)

def logthreshold(threshold_value, threshold, dic):
    ssim_percentage_threshold_dic = {}
    if threshold_value == threshold:
        ssim_percentage_threshold_dic = dic
    else:
        ssim_percentage_threshold_dic = None
    return ssim_percentage_threshold_dic


def countssim(tensor, imagesize):
    ssim_origin_dic = {}
    ssim_percentage_dic = {}

    filter_size_list = []
    threshold_value = 0
    max_ssim = 0.
    for filtersize in range(imagesize // 2 + 1):
        if filtersize == 0:
            _, image_fft = transfourier(tensor, 1, 1)
        else:
            _, image_fft = transfourier(tensor, 2 * filtersize, filtersize)
        # e^(-x)
        image_fft = torch.exp(-image_fft)
        # normalization
        image_fft = (image_fft-torch.mean(image_fft))/torch.std(image_fft)
        # compute the ssim
        ssim_value = ssim(tensor.cpu().numpy().astype(np.float32), image_fft.cpu().numpy().astype(np.float32),
                          data_range=1)
        ssim_value = round(ssim_value, 2)
        if filtersize == 0:
            max_ssim = ssim_value
            ssim_origin_dic[filtersize] = ssim_origin_dic.get(filtersize, 0) + ssim_value
            ssim_percentage_dic[filtersize] = ssim_percentage_dic.get(filtersize, 0) + ssim_value

        if ssim_value > 0:
            if filtersize != 0:
                filter_size_list.append(filtersize)
                ssim_origin_dic[filtersize] = ssim_origin_dic.get(filtersize, 0) + ssim_value
                ssim_percentage_dic[filtersize] = ssim_percentage_dic.get(filtersize, 0) + round(
                    (ssim_value / max_ssim), 3)
        else:
            threshold_value = filtersize - 1
            break

    return filter_size_list, threshold_value, ssim_origin_dic, ssim_percentage_dic


def updataexcel(excel_path, data_dic, column_number):
    new_data = pd.DataFrame([data_dic])
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        result = pd.concat([df, new_data], ignore_index=True)
        result.to_excel(excel_path, index=False, engine='openpyxl')

    except FileNotFoundError:
        dic_save = {}
        for column in range(column_number + 1):
            dic_save[column] = dic_save.get(column, [])
        df = pd.DataFrame(dic_save)
        print("Creat a new Excel file：", ssim_origin_excel_file)
        result = pd.concat([df, new_data], ignore_index=True)
        result.to_excel(excel_path, index=False, engine='openpyxl')


def updataexcel_list(excel_path, data_list, column_number):
    try:
        result = pd.read_excel(excel_path, engine='openpyxl')
    except FileNotFoundError:
        dic_save = {}
        for column in range(column_number + 1):
            dic_save[column] = dic_save.get(column, [])
        result = pd.DataFrame(dic_save)
    for dic in data_list:
        new_data = pd.DataFrame([dic])
        result = pd.concat([result, new_data], ignore_index=True)
    result.to_excel(excel_path, index=False, engine='openpyxl')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init the path
    ssim_origin_excel_file = 'the path where the ssim values are saved (file endwith .xlsx)'
    ssim_percentage_excel_file = 'the path where the percentage ssim values are saved (file endwith .xlsx)'
    ssim_percentage_single_excel_file = 'the path where the percentage ssim value of evey image are saved '
    path_threshold_value_txt = 'the path where the  statistical results for ssim values of each image are saved'

    # DATASET folder path
    imagenet_folder = "your dataset path"
    # target image size for training. ImageNet->(224,224) Agri17K->(224,224)
    target_size = (224, 224)

    # for accelerate, if your dataset is large, you can set the start_index and end_index to accelerate the process.
    # the code will start from the start_index and end at the end_index for save CPU memory and writting time of EXCEL file. 
    start_index = 0
    end_index = 1945685

    amplitude_counts = {}
    subdir_amplitude_counts = {}
    ssim_origin_list = []
    ssim_percentage_list = []

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    MaxPooling = torch.nn.MaxPool2d(kernel_size=4, stride=4)

    # main loop
    for root, dirs, files in os.walk(imagenet_folder):
        if root != imagenet_folder:
            if int(root.split('/')[-1].split('n')[-1]) >= start_index and  int(root.split('/')[-1].split('n')[-1]) <= end_index:
                print("current dir", root)
                for file in files:
                    if file.endswith(('.PNG', '.JPG', '.JPEG')):
                        image_path = os.path.join(root, file)
                        image = Image.open(image_path).convert('RGB') 

                        image_tensor = transform(image).to(device)
                        image_tensor = image_tensor.mean(0)
                        image_tensor = MaxPooling(image_tensor.unsqueeze(0).unsqueeze(0)).squeeze()

                        filter_size_list, threshold_value, ssim_origin_dic, ssim_percentage_dic = countssim(image_tensor, (target_size[0]//4))
                        ssim_origin_list.append(ssim_origin_dic)
                        ssim_percentage_list.append(ssim_percentage_dic)

                        amplitude_counts[threshold_value] = amplitude_counts.get(threshold_value, 0) + 1

                        subdir_name = os.path.dirname(image_path)
                        subdir_amplitude_counts[subdir_name] = subdir_amplitude_counts.get(subdir_name, {})
                        subdir_amplitude_counts[subdir_name][threshold_value] = subdir_amplitude_counts[subdir_name].get(
                            threshold_value, 0) + 1

                updataexcel_list(excel_path=ssim_origin_excel_file, data_list=ssim_origin_list,
                                column_number=target_size[0] // 2 + 1)

                updataexcel_list(excel_path=ssim_percentage_excel_file, data_list=ssim_percentage_list,
                                column_number=target_size[0] // 2 + 1)

                ssim_classifier = SSIMClass(size=target_size[0] // 4 + 1)
                for dic in ssim_percentage_list:
                    ssim_classifier.setvalue(dic)
                for i in range(1, target_size[0] // 4 + 1):
                    updataexcel_list(excel_path=ssim_percentage_single_excel_file + f'{i}.xlsx',
                                    data_list=getattr(ssim_classifier, f'ssim_percentage_{i + 1}'),
                                    column_number=target_size[0] // 4 + 1)

                ssim_origin_list.clear()
                ssim_percentage_list.clear()


    with open(path_threshold_value_txt, 'w') as file:
        for key, value in amplitude_counts.items():
            file.write(f'{key}: {value}\n')

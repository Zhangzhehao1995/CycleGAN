from argparse import ArgumentParser
import SimpleITK as sitk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys

# python C:\Users\zhehao.zhang\Desktop\CycleGAN_code\cyclegan_hugo\CycleGAN\convertdcm.py -i C:\Users\zhehao.zhang\Desktop\Newfolder\MR -o npy


def convert_from_dicom_to_jpg(img_array, save_path, index):
    save_jpg_path = os.path.join(save_path, str(index) + '.jpg')
    high_window = np.max(img_array)
    low_window = np.max([-1000, np.min(img_array)])
    win = np.array([low_window * 1., high_window * 1.])
    img_array = np.where(img_array < win[0], win[0], img_array)
    newImg = (img_array - win[0]) / (win[1] - win[0])  # 归一化
    newImg = (newImg * 255).astype('uint8')  # 将像素值扩展到[0,255]
    cv2.imwrite(save_jpg_path, newImg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def convert_from_dicom_to_npy(img_array, save_path, index):
    save_npy_path = os.path.join(save_path, str(index) + '.npy')
    np.save(save_npy_path, img_array)


def determine_type(file_name, form_type):
    if os.path.splitext(file_name)[1] == form_type:
        return True
    else:
        return False


if __name__ == '__main__':
    parser = ArgumentParser(description='Extra files names')
    parser.add_argument('-i', '--input_path', type=str, help='Define input path', required=True)
    parser.add_argument('-o', '--output_type', type=str, help='Npy or jpg', default='npy')
    parser.add_argument('-d', '--output_path', type=str, help='Output path')
    args = parser.parse_args()

    input_path = args.input_path
    if args.output_type == 'npy':
        convert_dcm = convert_from_dicom_to_npy
        output_path = input_path + '_npy'
    else:
        convert_dcm = convert_from_dicom_to_jpg
        output_path = input_path + '_jpg'

    if args.output_path is not None:
        output_path = args.output_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    index = 0
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if determine_type(file, '.dcm'):
                file_image_path = os.path.join(root, file)
                ds_array = sitk.ReadImage(file_image_path)  # 读取dicom文件的相关信息
                img_array = sitk.GetArrayFromImage(ds_array)  # 获取array
                # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高; shape类似于(1，height，width)
                shape = img_array.shape   # 获取array中的shape
                # print(shape)
                img_array = np.squeeze(img_array)  # 读取单张

                convert_dcm(img_array, output_path, index)  # 调用函数，转换成npy/jpg文件并保存到对应的路径
                index = index + 1

    print('Transformation complete')
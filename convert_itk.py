from argparse import ArgumentParser
import SimpleITK as sitk
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# python "D:\GAN code\cyclegan-cbct\convert_itk.py" -i C:\\Users\\PC\\Desktop\\test\\itk\\image.mha -o jpg -d C:\\Users\\PC\\Desktop\\test\\itk


def convert_from_itk_to_jpg(img_array, save_path, tag, low, high, normType = 0):
    save_jpg_path = os.path.join(save_path, tag + '.jpg')
    # normType: 0--3D norm; 1--2D norm
    if normType == 1:
        high = np.max(img_array)
        low = np.max([-1000, np.min(img_array)])
    win = np.array([low * 1., high * 1.])
    img_array = np.where(img_array < win[0], win[0], img_array)
    newImg = (img_array - win[0]) / (win[1] - win[0])  # 归一化
    newImg = (newImg * 255).astype('uint8')  # 将像素值扩展到[0,255]
    cv2.imwrite(save_jpg_path, newImg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def convert_from_itk_to_npy(img_array, save_path, tag, low, high, normType = 0):
    save_npy_path = os.path.join(save_path, tag + '.npy')
    if normType == 1:
        high = np.max(img_array)
        low = np.max([-1000, np.min(img_array)])
    win = np.array([low * 1., high * 1.])
    img_array = np.where(img_array < win[0], win[0], img_array)
    newImg = (img_array - win[0]) / (win[1] - win[0])  # 归一化
    np.save(save_npy_path, newImg)


def determine_type(file_name, form_type):
    if os.path.splitext(file_name)[1] == form_type:
        return True
    else:
        return False


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert ITK to npy/jpg')
    parser.add_argument('-i', '--input', type=str, help='Define input file path', required=True)
    parser.add_argument('-o', '--output_type', type=str, help='Npy or jpg', default='npy')
    parser.add_argument('-d', '--output_destination', type=str, help='Output destination', required=True)
    args = parser.parse_args()

    input_path = args.input
    output = args.output_destination
    if args.output_type == 'npy':
        convert_dcm = convert_from_itk_to_npy
    else:
        convert_dcm = convert_from_itk_to_jpg

    file_num = 0
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if determine_type(file, '.mha'):
                file_path = os.path.join(root, file)
                image3D = sitk.ReadImage(file_path)
                npSlice = sitk.GetArrayFromImage(image3D)
                high_3Dwindow = np.max(npSlice)
                low_3Dwindow = np.max([-1000, np.min(npSlice)])
                size = image3D.GetSize()
                newSize = list(size)
                newSize[2] = 0
                num = 0
                for z in range(size[2]):
                    index = (0, 0, z)
                    image2D = sitk.Extract(image3D, newSize, index)
                    slice2D = sitk.GetArrayFromImage(image2D)
                    tag = str(file_num) + '_' + str(num)
                    # last parameter 0/1 control 3D/2D normalization
                    convert_dcm(slice2D, output, tag, low_3Dwindow, high_3Dwindow, 0)  # 调用函数，转换成npy/jpg文件并保存到对应的路径
                    num = num + 1

                file_num = file_num + 1

    print('Complete...')


'''
npslice = sitk.GetArrayFromImage(image3D)
# print(npslice.shape)
origin = image3D.GetOrigin()
size = image3D.GetSize()
spacing = image3D.GetSpacing()
direction = image3D.GetDirection()
print(image3D.GetOrigin())
print(image3D.GetSpacing())
print(image3D.GetDirection())
# print(origin, size, spacing, direction)

newsize = list(image3D.GetSize())
newsize[2] = 0
results = []
for z in range(size[2]):
    index = (0, 0, z)
    # extractSliceFilter = sitk.ExtractImageFilter()
    # extractSliceFilter.SetSize(newsize)
    # extractSliceFilter.SetIndex(index)
    # image2D = extractSliceFilter.Execute(image3D)
    image2D = sitk.Extract(image3D, newsize, index)
    slice2D = sitk.GetArrayFromImage(image2D)
    # plt.imshow(slice2D)
    # plt.show()
    outslice = sitk.GetImageFromArray(slice2D)
    results.append(outslice)

outImg = sitk.JoinSeries(results)
outImg.CopyInformation(image3D)
print('----------------------------')
print(outImg.GetSize())
print(outImg.GetOrigin())
print(outImg.GetSpacing())
print(outImg.GetDirection())
sitk.WriteImage(outImg,outfile)
'''
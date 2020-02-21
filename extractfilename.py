from argparse import ArgumentParser
import os

# need to be called in Linux (otherwise, : can be used in linux but will be converted to _ in windows automatically)
# python cyclegan-cbct/extractfilename.py -d test_gan/data/vangogh2photo/images

def save_name_to_txt(dataset_path, image_dir):
    txt_name = os.path.join(dataset_path, image_dir + '.txt')
    f = open(txt_name, 'w')
    full_path = os.path.join(dataset_path, image_dir)
    filename_list = os.listdir(full_path)
    for i in range(len(filename_list)-1):  # 打印文件路径下的目录及文件名称
        f.write(os.path.splitext(filename_list[i])[0]+'\n')
    f.write(os.path.splitext(filename_list[-1])[0])
    f.close()


if __name__ == "__main__":

    parser = ArgumentParser(description='Extra files names')
    parser.add_argument('-d', '--data_path', type=str, help='Path to data', required=True)
    args = parser.parse_args()

    dataset_path = args.data_path
    dir_list = ['testA', 'testB', 'trainA', 'trainB']
    save_name_to_txt(dataset_path, dir_list[0])
    save_name_to_txt(dataset_path, dir_list[1])
    save_name_to_txt(dataset_path, dir_list[2])
    save_name_to_txt(dataset_path, dir_list[3])


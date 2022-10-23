import os
import glob
import random
import shutil
from PIL import Image
""" 对所有图片进行RGB转化，并且统一调整到一致大小，但不让图片发生变形或扭曲，划分了训练集和测试集 """

if __name__ == '__main__':
    test_split_ratio = 0.05
    desired_size = 128  # 图片缩放后的统一大小
    raw_path = './raw'

    dirs = glob.glob(os.path.join(raw_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]  # 只保留dirs中文件夹的部分，不考虑文件的部分

    print(f'Totally {len(dirs)} classes: {dirs}')

    for path in dirs:
        # 对每个类别单独处理

        path = path.split('\\')[-1]
        print(path)

        os.makedirs(f'train/{path}', exist_ok=True)
        os.makedirs(f'test/{path}', exist_ok=True)

        files = glob.glob(os.path.join(raw_path, path, '*'))  # 直接读取文件夹下的全部图片（前提是文件夹里的所有文件都是图片）
        #files += glob.glob(os.path.join(raw_path, path, '*.JPG'))
        #files += glob.glob(os.path.join(raw_path, path, '*.png'))

        random.shuffle(files)

        boundary = int(len(files)*test_split_ratio)  # 训练集和测试集的边界
        print('len(files):',len(files))
        print('boundary:',boundary)

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')

            old_size = img.size  # old_size[0] is in (width, height) format

            ratio = float(desired_size)/max(old_size)

            new_size = tuple([int(x*ratio) for x in old_size])

            im = img.resize(new_size, Image.ANTIALIAS)

            new_im = Image.new("RGB", (desired_size, desired_size))

            new_im.paste(im, ((desired_size-new_size[0])//2,
                                (desired_size-new_size[1])//2))

            assert new_im.mode == 'RGB'

            if i <= boundary:
                new_im.save(os.path.join(f'test/{path}', file.split('\\')[-1].split('.')[0]+'.jpg'))
            else:
                new_im.save(os.path.join(f'train/{path}', file.split('\\')[-1].split('.')[0]+'.jpg'))

    test_files = glob.glob(os.path.join('test', '*', '*.jpg'))
    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))

    print(f'Totally {len(train_files)} files for training')
    print(f'Totally {len(test_files)} files for test')
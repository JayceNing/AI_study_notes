import os 
import argparse
from matplotlib import pyplot as plt
import glob
import sys
import tqdm
import numpy as np
import torch

import torchvision
from PIL import Image
import timm
import clip

# 搜索相似图片的引擎

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args_parser():
    parser = argparse.ArgumentParser('image search task', add_help=False)

    # Model parameters

    parser.add_argument('--input_size', default=128, type=int,
                        help='images input size')

    parser.add_argument('--dataset_dir', default='./fruit_and_vegetable_data/train',
                        help='path where to load images')

    parser.add_argument('--test_image_dir', default='./fruit_and_vegetable_data/test_image',
                        help='images to test, split by comma "," ')

    parser.add_argument('--save_dir', default='./fruit_and_vegetable_data/output_dir',
                        help='path where to save, empty for no saving')

    parser.add_argument('--model_name', default='resnet50', # resnet50, resnet152, clip
                        help='model name')
    
    parser.add_argument('--feature_dict_file', default='corpus_feature_dict.npy',
                        help='filename where to save image representations')

    parser.add_argument('--topk', default=7, type=int,
                        help='k most similar images to be picked')

    parser.add_argument('--mode', default='extract', # extract or predict
                        help='extract or predict, for extraction features or prediction similar images from corpus')

    return parser

def extract_feature_single(args, model, file):

    img_rgb = Image.open(file).convert('RGB')
    image = img_rgb.resize((args.input_size, args.input_size), Image.ANTIALIAS)
    image = torchvision.transforms.ToTensor()(image)
    trainset_mean = [0.47083899, 0.43284143, 0.3242959]
    trainset_std = [0.37737389, 0.36130483, 0.34895992]
    image = torchvision.transforms.Normalize(mean=trainset_mean, std=trainset_std)(image).unsqueeze(0)

    with torch.no_grad():
        features = model.forward_features(image)
        vec = model.global_pool(features)
        vec = vec.squeeze().numpy()

    img_rgb.close()

    return vec

def extract_feature_by_CLIP(model, preprocess, file):
    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
        vec = vec.squeeze().cpu().numpy()
    
    return vec

def extract_features(args, model, image_path='', preprocess=None):

    allVectors = {}

    for image_file in tqdm.tqdm(glob.glob(os.path.join(image_path, '*', '*.jpg'))):
        if args.model_name == "clip":
            allVectors[image_file] = extract_feature_by_CLIP(model, preprocess, image_file)
        else:
            allVectors[image_file] = extract_feature_single(args, model, image_file)

    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)

    np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", allVectors)

    return allVectors

def getSimilarityMatrix(vectors_dict):

    v = np.array(list(vectors_dict.values()))  # [NUM, H]

    numerator = np.matmul(v, v.T)  # [NUM, NUM]
    denominator = np.matmul(np.linalg.norm(v, axis=1, keepdims=True), np.linalg.norm(v, axis=1, keepdims=True).T)  # [NUM, NUM]
    sim = numerator / denominator
    keys = list(vectors_dict.keys())

    return sim, keys

def setAxes(ax, image, query = False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query Image\n{0}".format(image), fontsize = 12)
        ax.xaxis.label.set_color('red')
    else:
        ax.set_xlabel("score={1:1.3f}\n{0}".format(image, value), fontsize = 12)
        ax.xaxis.label.set_color('blue')

    ax.set_xticks([])
    ax.set_yticks([])

def plotSimilarImages(args, image, simImages, simValues, numRow=1, numCol=4):
    fig = plt.figure()

    # set width and height in inches
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(f"use engine model: {args.model_name}", fontsize=35)

    for j in range(0, numCol*numRow):
        ax = []
        if j ==0:
            img = Image.open(image)
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, image.split(os.sep)[-1], query = True)
        else:
            img = Image.open(simImages[j-1])
            ax.append(fig.add_subplot(numRow, numCol, j+1))
            setAxes(ax[-1], simImages[j-1].split(os.sep)[-1], value = simValues[j-1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()

    fig.savefig(f"{args.save_dir}/{args.model_name}_search_top_{args.topk}_{image.split(os.sep)[-1].split('.')[0]}.png")
    plt.show()

if __name__ == '__main__':
    from pprint import pprint
    model_names = timm.list_models(pretrained=True)
    # pprint(model_names)

    args = get_args_parser()
    args = args.parse_args()

    model = None
    preprocess = None

    if args.model_name != "clip":
        model = timm.create_model(args.model_name, pretrained=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))
        model.eval()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

    if args.mode == "extract":
        # 第一阶段：图像表征提取
        print(f'use pretrained model {args.model_name} to extract features')
        allVectors = extract_features(args, model, image_path=args.dataset_dir, preprocess=preprocess)

    else:
        # 第二阶段：图像检索
        print(f'use pretrained model {args.model_name} to search {args.topk} similar images from corpus')

        test_images = glob.glob(os.path.join(args.test_image_dir, "*.png"))
        test_images += glob.glob(os.path.join(args.test_image_dir, "*.jpg"))
        test_images += glob.glob(os.path.join(args.test_image_dir, "*.jpeg"))

        # loading image representation dictionary
        allVectors = np.load(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", allow_pickle=True)
        allVectors = allVectors.item()

        # reading test images
        for image_file in tqdm.tqdm(test_images):
            print(f"reading {image_file}...")

            if args.model_name == "clip":
                # CLIP model
                allVectors[image_file] = extract_feature_by_CLIP(model, preprocess, image_file)
            else:
                # resnet50/152 model
                allVectors[image_file] = extract_feature_single(args, model, image_file)

        sim, keys = getSimilarityMatrix(allVectors)
        result = {}
        for image_file in tqdm.tqdm(test_images):
            print(f"sorting most similar images as {image_file}...")
            index = keys.index(image_file)
            sim_vec = sim[index]
            indexs = np.argsort(sim_vec)[::-1][1:args.topk]  # argsort 从小到大 [::-1] 翻转从大到小 [1:topk] 选取第一到topk个 刨去自己(1)
            simImages, simScores = [], []
            for ind in indexs:
                simImages.append(keys[ind])
                simScores.append(sim_vec[ind])
            result[image_file] = (simImages, simScores)

        print("starting to show similar images...")
        for image_file in test_images:
            plotSimilarImages(args, image_file, result[image_file][0], result[image_file][1], numRow=1, numCol=args.topk)
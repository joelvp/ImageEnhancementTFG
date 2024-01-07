from matplotlib import pyplot as plt
import numpy as np
import skimage
import skimage.io as io
from skimage.transform import rescale
import scipy.io as scio
import distortion_model
import argparse
import os
from PIL import Image
import random
from skimage import transform, util
import shutil
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, ToPILImage, RandomCrop, AutoAugment


# For parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sourcedir", type=str, default='C:/Users/JoelVP/Desktop/UPV/ImageEnhancementTFG/models/GeoProj/dataset/demo_images')
parser.add_argument("--datasetdir", type=str, default='C:/Users/JoelVP/Desktop/UPV/ImageEnhancementTFG/models/GeoProj/dataset/Data_gen')
parser.add_argument("--trainnum", type=int, default=12, help='number of the training set')
parser.add_argument("--testnum", type=int, default=3, help='number of the test set')
parser.add_argument("--data_augmentation", action='store_false', help='augmentation data')
parser.add_argument("--augment_factor", type=int, default=2, help='augmentation factor for data')
parser.add_argument("--data_rename", type=bool, default=True, help='rename data')
parser.add_argument("--data_resize", type=bool, default=True, help='resize data')

args = parser.parse_args()

if not os.path.exists(args.datasetdir):
    os.mkdir(args.datasetdir)

trainDisPath = args.datasetdir + '/train_distorted'
trainUvPath = args.datasetdir + '/train_flow'
testDisPath = args.datasetdir + '/test_distorted'
testUvPath = args.datasetdir + '/test_flow'

if not os.path.exists(trainDisPath):
    os.mkdir(trainDisPath)

if not os.path.exists(trainUvPath):
    os.mkdir(trainUvPath)

if not os.path.exists(testDisPath):
    os.mkdir(testDisPath)

if not os.path.exists(testUvPath):
    os.mkdir(testUvPath)


def augment_data(input_folder, output_folder, augment_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [file for file in os.listdir(input_folder) if
                   file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    transform_auto = Compose([
        ToTensor(),
        AutoAugment(),
    ])
    transform = Compose([
        ToTensor(),
        RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2), antialias=True),
        #RandomHorizontalFlip(p=0.5),
        
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        original_image = Image.open(input_image_path)


        for i in range(augment_factor):
            augmented_image = augment_image(original_image, transform)

            # Convertir el tensor resultante en una imagen PIL
            augmented_image_pil = ToPILImage()(augmented_image)

            augmented_image_path = os.path.join(output_folder, f'{image_file[:-4]}_{i:02d}.jpg')

            augmented_image_pil.save(augmented_image_path)

def augment_image(image, transform):
    # Apply the provided torchvision transform
    augmented_image = transform(image)

    return augmented_image


def rename_images(folder_path):
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    image_files.sort()
    for index, old_name in enumerate(image_files):
        new_name = '{:06d}.jpg'.format(index)
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f'Renamed: {old_name} -> {new_name}')


def crop_and_resize(image_path, target_size):
    img = Image.open(image_path)
    original_width, original_height = img.size
    min_dim = min(original_width, original_height)
    left = (original_width - min_dim) // 2
    top = (original_height - min_dim) // 2
    right = (original_width + min_dim) // 2
    bottom = (original_height + min_dim) // 2
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((target_size, target_size), Image.ANTIALIAS)
    return resized_img


def process_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [file for file in os.listdir(input_folder) if
                   file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        resized_and_cropped_img = crop_and_resize(input_image_path, target_size)
        output_image_path = os.path.join(output_folder, image_file)
        print('##################')
        print(resized_and_cropped_img)
        print('##################')
        resized_and_cropped_img.save(output_image_path)
        print(f'Processed: {input_image_path} -> {output_image_path}')


def generatedata(types, k, trainFlag):
    print(types, trainFlag, k)

    width = 512
    height = 512

    parameters = distortion_model.distortionParameter(types)

    OriImg = io.imread('%s%s%s%s' % (processed_folder, '/', str(k).zfill(6), '.jpg'))

    disImg = np.array(np.zeros(OriImg.shape), dtype=np.uint8)
    u = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)
    v = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)

    cropImg = np.array(np.zeros((int(height / 2), int(width / 2), 3)), dtype=np.uint8)
    crop_u = np.array(np.zeros((int(height / 2), int(width / 2))), dtype=np.float32)
    crop_v = np.array(np.zeros((int(height / 2), int(width / 2))), dtype=np.float32)

    xmin = int(width * 1 / 4)
    xmax = int(width * 3 / 4 - 1)
    ymin = int(height * 1 / 4)
    ymax = int(height * 3 / 4 - 1)

    for i in range(width):
        for j in range(height):

            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)

            if (0 <= xu < width - 1) and (0 <= yu < height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j

                Q11 = OriImg[int(yu), int(xu), :]
                Q12 = OriImg[int(yu), int(xu) + 1, :]
                Q21 = OriImg[int(yu) + 1, int(xu), :]
                Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]

                disImg[j, i, :] = Q11 * (int(xu) + 1 - xu) * (int(yu) + 1 - yu) + \
                                  Q12 * (xu - int(xu)) * (int(yu) + 1 - yu) + \
                                  Q21 * (int(xu) + 1 - xu) * (yu - int(yu)) + \
                                  Q22 * (xu - int(xu)) * (yu - int(yu))

                if (xmin <= i <= xmax) and (ymin <= j <= ymax):
                    cropImg[j - ymin, i - xmin, :] = disImg[j, i, :]
                    crop_u[j - ymin, i - xmin] = u[j, i]
                    crop_v[j - ymin, i - xmin] = v[j, i]

    if trainFlag:
        saveImgPath = '%s%s%s%s%s%s' % (trainDisPath, '/', types, '_', str(k).zfill(6), '.jpg')
        saveMatPath = '%s%s%s%s%s%s' % (trainUvPath, '/', types, '_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})
    else:
        saveImgPath = '%s%s%s%s%s%s' % (testDisPath, '/', types, '_', str(k).zfill(6), '.jpg')
        saveMatPath = '%s%s%s%s%s%s' % (testUvPath, '/', types, '_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})

##### Process folder #####
# Crear una carpeta procesada basada en la carpeta fuente
processed_folder = args.sourcedir + "_processed"

if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
    # Copiar todas las imágenes de la carpeta fuente a la carpeta procesada
    image_files = [file for file in os.listdir(args.sourcedir) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    for image_file in image_files:
        source_path = os.path.join(args.sourcedir, image_file)
        destination_path = os.path.join(processed_folder, image_file)
        shutil.copyfile(source_path, destination_path)

##### Augment Data #####

if args.data_augmentation:
    augment_data(processed_folder, processed_folder, args.augment_factor)

##### Rename Images #####
if args.data_rename:
    rename_images(processed_folder)

##### Process Images #####
# Ruta de la carpeta de entrada, carpeta de salida y tamaño objetivo
target_size = 512

# Llamar a la función para procesar todas las imágenes en la carpeta de entrada para cropear y resize
if args.data_resize:
    process_images(processed_folder, processed_folder, target_size)

# Only barrel
for types in ['barrel']:
    for k in range(args.trainnum):
        generatedata(types, k, trainFlag=True)

    for k in range(args.trainnum, args.trainnum + args.testnum):
        generatedata(types, k, trainFlag=False)

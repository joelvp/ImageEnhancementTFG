import logging
import random
import numpy as np
import skimage.io as io
import scipy.io as scio
import distortion_model
import argparse
import os
from PIL import Image
import torchvision.transforms as T 
import shutil

# For parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sourcedir", type=str, default='C:/Users/JoelVP/Desktop/UPV/ImageEnhancementTFG/models/GeoProj/dataset/Places25k')
parser.add_argument("--datasetdir", type=str, default='C:/Users/JoelVP/Desktop/UPV/ImageEnhancementTFG/models/GeoProj/dataset/Data_gen')
parser.add_argument("--trainnum", type=int, default=20000, help='number of the training set')
parser.add_argument("--testnum", type=int, default=2000, help='number of the test set')
parser.add_argument("--data_augmentation", action='store_true', help='augmentation data')
parser.add_argument("--augment_factor", type=int, default=3, help='augmentation factor for data')

# Transformations for data augmentation
transform_auto = T.Compose([
        T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
    ])
transform_basic = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-45,45)),
])

# Functions
def transform_image(image, transform):
    # Apply the provided torchvision transform
    augmented_image = transform(image)

    return augmented_image
def augment_data(folders, augment_factor, transform):

    image_files = [file for file in os.listdir(folders['processedFolder']) if
                   file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for image_file in image_files:
        input_image_path = os.path.join(folders['processedFolder'], image_file)
        original_image = Image.open(input_image_path)

        for i in range(augment_factor):
            augmented_image = transform_image(original_image, transform)

            augmented_image_path = os.path.join(folders['processedFolder'], f'{image_file[:-4]}_{i:02d}.jpg')

            augmented_image.save(augmented_image_path)

def rename_and_shuffle_images(folders):
    files = os.listdir(folders['processedFolder'])
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Mezclar el orden de las imágenes de manera aleatoria
    random.shuffle(image_files)

    # Cambiar el nombre de las imágenes en orden
    for index, old_name in enumerate(image_files):
        new_name = '{:06d}.jpg'.format(index)
        old_path = os.path.join(folders['processedFolder'], old_name)
        new_path = os.path.join(folders['processedFolder'], new_name)
        os.rename(old_path, new_path)
        logging.info(f'Renamed and shuffled: {old_name} -> {new_name}')

def generatedata(types, folders, img_width_dataset, img_height_dataset, k, trainFlag):
    logging.info(f"Types: {types}, Train Flag: {trainFlag}, K: {k}")

    #Size of the original image to be distorted
    width = img_width_dataset * 2
    height = img_height_dataset * 2

    parameters = distortion_model.distortionParameter(types, img_width_dataset, img_height_dataset)

    OriImg = io.imread('%s%s%s%s' % (folders['processedFolder'], '/', str(k).zfill(6), '.jpg'))
    
    # Calculate start and end coordinates for the centred cutout
    crop_x_start = (OriImg.shape[1] - width) // 2
    crop_x_end = crop_x_start + width
    crop_y_start = (OriImg.shape[0] - height) // 2
    crop_y_end = crop_y_start + height
    
    # Create the centred cropped image
    OriImg = OriImg[crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]

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

                # New pixel value applying distortion
                disImg[j, i, :] = Q11 * (int(xu) + 1 - xu) * (int(yu) + 1 - yu) + \
                                  Q12 * (xu - int(xu)) * (int(yu) + 1 - yu) + \
                                  Q21 * (int(xu) + 1 - xu) * (yu - int(yu)) + \
                                  Q22 * (xu - int(xu)) * (yu - int(yu))

                # Final image always half of the image distorted, to eliminate black borders.s
                if (xmin <= i <= xmax) and (ymin <= j <= ymax):
                    cropImg[j - ymin, i - xmin, :] = disImg[j, i, :]
                    crop_u[j - ymin, i - xmin] = u[j, i]
                    crop_v[j - ymin, i - xmin] = v[j, i]

    if trainFlag:
        saveImgPath = '%s%s%s%s%s%s' % (folders['trainDistorted'], '/', types, '_', str(k).zfill(6), '.jpg')
        saveMatPath = '%s%s%s%s%s%s' % (folders['trainFlow'], '/', types, '_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})
    else:
        saveImgPath = '%s%s%s%s%s%s' % (folders['testDistorted'], '/', types, '_', str(k).zfill(6), '.jpg')
        saveMatPath = '%s%s%s%s%s%s' % (folders['testFlow'], '/', types, '_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})
        
def filter_size_images(folders, img_width_dataset, img_height_dataset):
    
    min_width = img_width_dataset * 2  # Sets the desired minimum width value
    min_height = img_height_dataset * 2  # Sets the desired minimum height value

    # Copy only images that comply with the width and height requirements
    image_files = [file for file in os.listdir(folders['sourcedir']) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    for image_file in image_files:
        source_path = os.path.join(folders['sourcedir'], image_file)
        destination_path = os.path.join(folders['processedFolder'], image_file)

        # Obtain image dimensions
        image = Image.open(source_path)
        image_width, image_height = image.size

        # Check if it meets the minimum width and height requirements
        # and if it's a color image (not grayscale)
        if image_width >= min_width and image_height >= min_height and image.mode != 'L':
            shutil.copyfile(source_path, destination_path)
            
def folder_creations(args):
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
        
    processed_folder = args.sourcedir + "_processed"
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        
    return {
        'sourcedir': args.sourcedir,
        'processedFolder': processed_folder,
        'datasetdir': args.datasetdir,
        'trainDistorted': trainDisPath,
        'trainFlow': trainUvPath,
        'testDistorted': testDisPath,
        'testFlow': testUvPath
        
    }
    

##### MAIN #####
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    args = parser.parse_args()
    
    folders = folder_creations(args)
    
    img_width_dataset = 256
    img_height_dataset = 256
    
    rename = False # Cambiar a False
        
    ##### Filter Size Images #####
    if not os.listdir(folders['processedFolder']):
        filter_size_images(folders, img_width_dataset, img_height_dataset)
        rename = True
    ##### Augment Data #####
    if args.data_augmentation:
        augment_data(folders, args.augment_factor, transform_basic)

    ##### Rename and shuffle Images #####
    if rename:
        rename_and_shuffle_images(folders)

    # Only barrel
    for types in ['barrel']:
        for k in range(19625, args.trainnum):
            generatedata(types,folders,img_width_dataset, img_height_dataset, k, trainFlag=True)

        for k in range(args.trainnum, args.trainnum + args.testnum):
            generatedata(types, folders, img_width_dataset, img_height_dataset, k, trainFlag=False)
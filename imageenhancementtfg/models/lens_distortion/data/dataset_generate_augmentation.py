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
parser.add_argument("--sourcedir", type=str, default='C:/Users/JoelVP/Desktop/UPV/mini_dataset')
parser.add_argument("--datasetdir", type=str, default='C:/Users/JoelVP/Desktop/UPV/ImageEnhancementTFG/imageenhancementtfg/models/lens_distortion/dataset/demo')
parser.add_argument("--train_percentage", type=float, default=0.6, help='percentage of images for training (between 0 and 1)')
parser.add_argument("--test_percentage", type=float, default=0.4, help='percentage of images for testing (between 0 and 1)')
parser.add_argument("--data_augmentation", action='store_false', help='augmentation data')
parser.add_argument("--augment_factor", type=int, default=3, help='augmentation factor for data')

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Transformations for data augmentation
transform_auto = T.Compose([
        T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
    ])
transform_basic = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-45, 45)),
])

# Constants for dataset preprocessing
IMG_WIDTH_DATASET = 256
IMG_HEIGHT_DATASET = 256


# Functions
def transform_image(image: Image.Image, transform: T.Compose) -> Image.Image:
    """
    Apply a torchvision transform to an image.

    Parameters:
        image (Image.Image): Input image.
        transform (T.Compose): Torchvision transform to apply.

    Returns:
        Image.Image: Transformed image.
    """
    augmented_image = transform(image)

    return augmented_image


def augment_data(folders: dict, augment_factor: int, transform: T.Compose):
    """
    Augment data by applying transformations to images.

    Parameters:
        folders (dict): Dictionary containing folder paths.
        augment_factor (int): Number of augmented images per original image.
        transform (T.Compose): Torchvision transform for augmentation.
    """

    image_files = [file for file in os.listdir(folders['processedFolder']) if
                   file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for image_file in image_files:
        input_image_path = os.path.join(folders['processedFolder'], image_file)
        original_image = Image.open(input_image_path)

        for i in range(augment_factor):
            augmented_image = transform_image(original_image, transform)
            augmented_image_path = os.path.join(folders['processedFolder'], f'{image_file[:-4]}_{i:02d}.jpg')
            augmented_image.save(augmented_image_path)
            
            
def rename_and_shuffle_images(folder: dict):
    """
    Rename and shuffle images in the processed folder.

    Parameters:
        folder (dict): Dictionary containing folder paths.
    """
    files = os.listdir(folder['processedFolder'])
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    random.shuffle(image_files)

    for index, old_name in enumerate(image_files):
        new_name = '{:06d}.jpg'.format(index)
        old_path = os.path.join(folder['processedFolder'], old_name)
        new_path = os.path.join(folder['processedFolder'], new_name)
        os.rename(old_path, new_path)
        logging.info(f'Renamed and shuffled: {old_name} -> {new_name}')
        
        
def generatedata(types: str, folders: dict, img_width_dataset: int, img_height_dataset: int, k: int, train_flag: bool):
    """
    Generate distorted images and save them to respective folders.

    Parameters:
        types (str): Type of distortion.
        folders (dict): Dictionary containing folder paths.
        img_width_dataset (int): Width of the dataset images.
        img_height_dataset (int): Height of the dataset images.
        k (int): Index of the image.
        train_flag (bool): Flag indicating if image is for training or testing.
    """
    logging.info(f"Generating data for type: {types}, Train Flag: {train_flag}, Index: {k}")

    width = img_width_dataset * 2
    height = img_height_dataset * 2

    parameters = distortion_model.distortionParameter(types, img_width_dataset, img_height_dataset)
    OriImg = io.imread(os.path.join(folders['processedFolder'], f'{k:06d}.jpg'))
    
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

    if train_flag:
        saveImgPath = f"{folders['trainDistorted']}/{types}_{k:06d}.jpg"
        saveMatPath = f"{folders['trainFlow']}/{types}_{k:06d}.mat"
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})
    else:
        saveImgPath = f"{folders['testDistorted']}/{types}_{k:06d}.jpg"
        saveMatPath = f"{folders['testFlow']}/{types}_{k:06d}.mat"
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})


def filter_images(folders: dict, img_width_dataset: int, img_height_dataset: int):
    """
    Filter and copy images from source directory to processed folder based on size criteria.

    Parameters:
        folders (dict): Dictionary containing folder paths.
        img_width_dataset (int): Width of the dataset images.
        img_height_dataset (int): Height of the dataset images.
    """
    
    min_width = img_width_dataset * 2  # Sets the desired minimum width value
    min_height = img_height_dataset * 2  # Sets the desired minimum height value

    image_files = [file for file in os.listdir(folders['sourcedir']) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    for image_file in image_files:
        source_path = os.path.join(folders['sourcedir'], image_file)
        destination_path = os.path.join(folders['processedFolder'], image_file)

        image = Image.open(source_path)
        image_width, image_height = image.size

        if image_width >= min_width and image_height >= min_height and image.mode != 'L':
            shutil.copyfile(source_path, destination_path)


def create_folders(args: argparse.Namespace) -> dict:
    """
    Create necessary folders for dataset generation.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary containing folder paths.
    """
    if not os.path.exists(args.datasetdir):
        os.mkdir(args.datasetdir)

    for folder in ['trainDistorted', 'trainFlow', 'testDistorted', 'testFlow']:
        path = os.path.join(args.datasetdir, folder)
        if not os.path.exists(path):
            os.mkdir(path)
        
    processed_folder = args.sourcedir + "_processed"
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    return {
        'sourcedir': args.sourcedir,
        'processedFolder': processed_folder,
        'datasetdir': args.datasetdir,
        'trainDistorted': os.path.join(args.datasetdir, 'trainDistorted'),
        'trainFlow': os.path.join(args.datasetdir, 'trainFlow'),
        'testDistorted': os.path.join(args.datasetdir, 'testDistorted'),
        'testFlow': os.path.join(args.datasetdir, 'testFlow')
        
    }


def prepare_indexes_started_generation(folders: dict, args: argparse.Namespace) -> tuple:
    """
    Prepare start indexes for generating data when previous data exists.

    Parameters:
        folders (dict): Dictionary containing folder paths.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: Train and test index ranges.
    """
    train_img_folder = folders['trainDistorted']
    test_img_folder = folders['testDistorted']
    train_flow_folder = folders['trainFlow']
    test_flow_folder = folders['testFlow']

    # Calculate total number of images after augmentation
    image_files = [file for file in os.listdir(folders['processedFolder']) if
                   file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    total_images = len(image_files)

    # Calculate trainnum and testnum based on percentages
    trainnum_total = int(total_images * args.train_percentage)
    testnum_total = total_images

    trainnum_idx = 0
    testnum_idx = trainnum_total

    # Check train folder for existing images and determine start index
    if os.path.exists(train_img_folder) and os.listdir(train_img_folder):
        # Get the last index in train images.
        train_img_files = [file for file in os.listdir(train_img_folder) if
                           file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        if train_img_files:
            trainnum_idx = max([int(filename.split('_')[-1].split('.')[0]) for filename in train_img_files]) + 1

        if os.path.exists(train_flow_folder) and os.listdir(train_flow_folder):
            # Double check if is the same index as in train images, if is different get the lowest one
            train_flow_files = [file for file in os.listdir(train_flow_folder) if
                                file.lower().endswith(('.mat'))]
            if train_flow_files:
                trainnum_flow_idx = max(
                    [int(filename.split('_')[-1].split('.')[0]) for filename in train_flow_files]) + 1
                trainnum_idx = min(trainnum_idx, trainnum_flow_idx)

                if trainnum_idx < trainnum_total:
                    return trainnum_total, testnum_total, trainnum_idx, testnum_idx


        # Check test folder for existing images and determine start index
        if os.path.exists(test_img_folder) and os.listdir(test_img_folder):
            # Get the last index in test images.
            test_img_files = [file for file in os.listdir(test_img_folder) if
                              file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            if test_img_files:
                testnum_idx = max([int(filename.split('_')[-1].split('.')[0]) for filename in test_img_files]) + 1

            if os.path.exists(test_flow_folder) and os.listdir(test_flow_folder):
                # Double check if is the same index as in test images, if is different get the lowest one
                test_flow_files = [file for file in os.listdir(test_flow_folder) if
                                   file.lower().endswith(('.mat'))]
                if test_flow_files:
                    testnum_flow_idx = max(
                        [int(filename.split('_')[-1].split('.')[0]) for filename in test_flow_files]) + 1
                    testnum_idx = min(testnum_idx, testnum_flow_idx)

    return trainnum_total, testnum_total, trainnum_idx, testnum_idx


def prepare_indexes_new_generation(folders: dict) -> tuple:
    """
    Prepare start indexes for generating data when no previous data exists.

    Parameters:
        folders (dict): Dictionary containing folder paths.

    Returns:
        tuple: Train and test index ranges.
    """
    # Calculate total number of images after augmentation
    image_files = [file for file in os.listdir(folders['processedFolder']) if
                   file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    total_images = len(image_files)

    # Calculate trainnum and testnum based on percentages
    trainnum_total = int(total_images * args.train_percentage)
    testnum_total = total_images

    trainnum_idx = 0
    testnum_idx = trainnum_total

    return trainnum_total, testnum_total, trainnum_idx, testnum_idx



##### MAIN #####
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    args = parser.parse_args()
    folders = create_folders(args)

    if not os.listdir(folders['processedFolder']): # If empty processed folder with dataset is new generation
        filter_images(folders, IMG_WIDTH_DATASET, IMG_HEIGHT_DATASET)

        ##### Augment Data #####
        if args.data_augmentation:
            print("Hey")
            augment_data(folders, args.augment_factor, transform_basic)

        ##### Rename and shuffle Images #####
        rename_and_shuffle_images(folders)

        ##### Prepare indexes #####
        trainnum_total, testnum_total, trainnum_idx, testnum_idx = prepare_indexes_new_generation(folders)

    else: # If processed folder with dataset exists, then check how many final images are there
        trainnum_total, testnum_total, trainnum_idx, testnum_idx = prepare_indexes_started_generation(folders, args)

    # Only barrel
    for types in ['barrel']:
        for k in range(trainnum_idx, trainnum_total):
            generatedata(types,folders,IMG_WIDTH_DATASET, IMG_HEIGHT_DATASET, k, train_flag=True)

        for k in range(testnum_idx, testnum_total):
            generatedata(types, folders, IMG_WIDTH_DATASET, IMG_HEIGHT_DATASET, k, train_flag=False)

import tensorflow as tf
from PIL import Image
import numpy as np
from torchvision import transforms
import logging

from models.utils import load_config

config = load_config('data/config.ini')


def load_esrgan_model(model_path: str) -> tf.keras.Model:
    """
    Load the Enhanced Super Resolution GAN (ESRGAN) model from TensorFlow.

    Parameters:
        model_path (str): Path to the saved model directory.

    Returns:
        tf.keras.Model: Loaded ESRGAN model.
    """
    model = tf.saved_model.load(model_path)
    return model


def preprocessing(img: np.ndarray) -> tf.Tensor:
    """
    Preprocess the image for the ESRGAN model.

    Parameters:
        img (np.ndarray: Input image numpy  .

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    image_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    cropped_image = tf.image.crop_to_bounding_box(img, 0, 0, image_size[0], image_size[1])
    preprocessed_image = tf.cast(cropped_image, tf.float32)
    return tf.expand_dims(preprocessed_image, 0)


def super_resolution(input_image: np.ndarray) -> np.ndarray:
    """
    Perform super resolution on the input image using ESRGAN.

    Parameters:
        input_image (np.ndarray): Input image as numpy array.

    Returns:
        np.ndarray: Super-resolved image as numpy array.
    """

    # Load ESRGAN model
    esrgan_model = load_esrgan_model(config['models']['super_resolution_model'])

    # Perform super resolution
    preprocessed_image = preprocessing(input_image)
    enhanced_image_tf = esrgan_model(preprocessed_image)
    enhanced_image_np = tf.squeeze(enhanced_image_tf).numpy().astype(np.uint8)
    logging.info(f"Super resolution successful")

    return enhanced_image_np


def transform_image(image: np.ndarray, device: str) -> tuple:
    """
    Transform the input image into tensors suitable for deep learning models.

    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        device (str): Device to use for tensor operations ('cpu' or 'gpu')

    Returns:
        tuple: Tuple containing transformed image as tensor and NumPy array.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    im = Image.fromarray(image)  # Convert numpy array to PIL Image

    # Crop to square while preserving the larger dimension
    width, height = im.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    im = im.crop((left, top, right, bottom))

    im_npy = np.asarray(im.resize((256, 256)))
    # Convert to tensor and normalize
    im_tensor = transform(im).unsqueeze(0).to(device)
    return im_tensor, im_npy
import torch
import logging
import gradio as gr
import configparser

gradio_flag = False


def load_config(path: str) -> configparser.ConfigParser:
    """
    Load the configuration file.

    Parameters:
        path (str): Path to the configuration file.

    Returns:
        configparser.ConfigParser: Loaded configuration.
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config


def clear_cuda_cache(empty_cuda: bool = True) -> None:
    """
    Clear the CUDA cache if a CUDA-enabled GPU is available.

    If the `gradio_flag` is set to True, it will display an info message using Gradio.
    The `empty_cuda` parameter is included for compatibility with tenacity retry mechanisms.

    Parameters
    ----------
    empty_cuda : bool, optional
        Placeholder parameter for retry compatibility, by default True.

    Returns
    -------
    None
    """
    global gradio_flag

    if torch.cuda.is_available():
        if gradio_flag:
            gr.Info("Trying to empty CUDA cache")
        torch.cuda.empty_cache()
        logging.info("CUDA cache cleared")

        gradio_flag = True


def reset_gradio_flag() -> None:
    """
    Reset the global flag `gradio_flag` to False.

    This function marks that `clear_cuda_cache` has been called at least once.

    Returns
    -------
    None
    """
    global gradio_flag
    gradio_flag = False
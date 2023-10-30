import argparse
import logging
from models.Deep_White_Balance.PyTorch.white_balance import white_balance


def get_args():
    parser = argparse.ArgumentParser(description='Image Enhancement.')
    parser.add_argument('--WB', action='store_true',
                        help="Applying automatic WB")
    parser.add_argument('--lowlight', action='store_true',
                        help="Low-light enhancement")
    
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    wb = args.WB
    llight = args.lowlight
    
    if wb:
        white_balance()
    if llight:
        white_balance()
    else:
        raise Exception('No enhancement applied!')

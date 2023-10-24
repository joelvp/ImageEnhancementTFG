import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Image Enhancement.')
    parser.add_argument('--WB', default=False,
                        help="Applying automatic WB")
    
    return parser.parse_args()

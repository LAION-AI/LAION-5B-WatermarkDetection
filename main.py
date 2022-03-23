import torch
import argparse
from inference import inference

def main():
    """
    Launch inference using torch spawn with number of processes equal to the number of devices.
    """
    args = parse_args()
    torch.multiprocessing.spawn(inference, nprocs=torch.cuda.device_count(), args=(args,))


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', type=str, help='file urls')
    parser.add_argument('--num-samples', type=int, help='number of pairs')
    parser.add_argument('--batch-size', type=int, help='samples per batch')
    return parser.parse_args()

if __name__ == '__main__':
    main()
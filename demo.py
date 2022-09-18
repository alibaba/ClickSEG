import argparse
import tkinter as tk

import torch

from isegm.utils import exp
from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp


def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    checkpoint_path = args.checkpoint
    model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)

    root = tk.Tk()
    root.minsize(960, 960)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Absolute path to the checkpoint.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=960,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')

    return args


if __name__ == '__main__':
    main()

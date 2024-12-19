from utils.data_utils import get_loader
import argparse
import os

from tqdm import tqdm

import jittor as jt
from models.modeling import VisionTransformer, CONFIGS
jt.flags.use_cuda = 1





parser = argparse.ArgumentParser()

parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011", help="Which dataset.")

parser.add_argument('--data_root', type=str, default='./minist')

parser.add_argument("--train_batch_size", default=2, type=int, help="Total batch size for training.")

parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")

parser.add_argument("--img_size", default=448, type=int, help="Resolution size")

parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value\n")

args = parser.parse_args()
args.data_root = os.path.join(args.data_root, args.dataset)
train_loader, test_loader = get_loader(args)


config = CONFIGS['debug']


if args.dataset == "CUB_200_2011":
    num_classes = 200
elif args.dataset == "car":
    num_classes = 196
elif args.dataset == "nabirds":
    num_classes = 555
elif args.dataset == "dog":
    num_classes = 120
elif args.dataset == "INat2017":
    num_classes = 5089

model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
while True:
    model.train()
    epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
            batch = tuple(batch)
            x, y = batch
            loss, logits = model(x, y)
            print(loss.shape, logits.shape)
    break
    
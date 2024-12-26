from utils.data_utils import get_loader
import argparse
import os

import logging
import time
from tqdm import tqdm
import numpy as np
import jittor as jt
from models.modeling import VisionTransformer, CONFIGS
jt.flags.use_cuda = 1


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def valid(args, model, test_loader, global_step):
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

    loss_fct = jt.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(batch)
        x, y = batch
        with jt.no_grad():
            logits = model(x)
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            
            preds, _ = jt.argmax(logits, dim=-1)
            
            if len(all_preds) == 0:
                all_preds.append(preds.detach().numpy())
                all_label.append(y.detach().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().numpy(), axis=0
                )
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_loss.detach().numpy())
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    
    print("")
    print("**********Validation Results**********")
    print("Global Steps: %d" % global_step)
    print("Valid Loss: %2.5f" % eval_loss.detach().numpy())
    print("Valid Accuracy: %2.5f" % accuracy)
    
    return accuracy


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011", help="Which dataset.")

parser.add_argument('--data_root', type=str, default='./minist')

parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")

parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")

parser.add_argument("--img_size", default=448, type=int, help="Resolution size")

parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value\n")

parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD.")

parser.add_argument("--num_steps", default=10001, type=int, help="Total number of training epochs to perform.")

parser.add_argument("--eval_every", default=2000, type=int, help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

parser.add_argument("--pretrained_dir", type=str, default="./minist/ViT-B_16.npz", help="Where to search for pretrained ViT models.")

parser.add_argument("--output_dir", type=str, default="./output", help="The output directory where the model checkpoints will be written.")

parser.add_argument("--contrastive", type=int, default=1, help="Whether to use contrastive learning. 0 for False, 1 for True.")

parser.add_argument('--split', type=str, default='overlap', help="Split method")

parser.add_argument('--margin', type=float, default=0.4, help="Margin for contrastive loss")

args = parser.parse_args()
args.data_root = os.path.join(args.data_root, args.dataset)
train_loader, test_loader = get_loader(args)


config = CONFIGS['ViT-B_16']
config.split = args.split

os.makedirs(args.output_dir, exist_ok=True)
filename = args.split + "_contrastive(" + str(args.contrastive) + ")_margin(" + str(args.margin) + ").txt"
filepath = os.path.join(args.output_dir, filename)

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

model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value, constrastive=args.contrastive, margin=args.margin)
model.load_from(np.load(args.pretrained_dir))
optimizer = jt.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
t_total = args.num_steps
scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_total)
global_step = 0
start_time = time.time()

while True:
    model.train()
    epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    all_preds, all_label = [], []
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(batch)
        x, y = batch
        loss, logits = model(x, y)
        loss = loss.mean()
        global_step += 1
        preds, _ = jt.argmax(logits, dim=-1)
        
        if len(all_preds) == 0:
            all_preds.append(preds.detach().numpy())
            all_label.append(y.detach().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().numpy(), axis=0
            )
        optimizer.step(loss)
        scheduler.step()
        
        epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.detach().numpy())
                )
        
        if global_step % args.eval_every == 0:
            log_time = time.time()
            accuracy = valid(args, model, test_loader, global_step)
            valid_time = time.time() - log_time
            if jt.rank == 0:
                with open(filepath, "a+") as f:
                    f.write(f"Train Duration: {log_time - start_time}s Step: {global_step}, Accuracy: {accuracy}\n")
            
            start_time += valid_time
        
        if global_step >= t_total:
            break
    
    all_preds, all_label = all_preds[0], all_label[0]
    train_accuracy = simple_accuracy(all_preds, all_label)
    
    if global_step >= t_total:
        break
    
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "......"
from argparse import ArgumentParser
import torch

torch.backends.cudnn.benchmark = True
import yaml
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from utils import Kinetics400
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, SGD, Adagrad,Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils import load_yaml, GradualWarmupScheduler
from model import VTN
from utils.Tree_Loss import TreeLoss
from torchmetrics import AUROC, Specificity, Recall, F1Score, Accuracy
from torchvision.io.video import read_video
import pandas as pd

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--annotations", type=str, default="./*/*/train.txt",
                    help="Dataset labels path")
parser.add_argument("--val-annotations", type=str, default="./*/*/val.txt",
                    help="Dataset labels path")
parser.add_argument("--root-dir", type=str, default="/*/train/video",
                    help="Dataset files root-dir")
parser.add_argument("--val-root-dir", type=str, default="/*/val/video",
                    help="Dataset files root-dir")
parser.add_argument("--classes", type=int, default=4, help="Number of classes")
parser.add_argument("--config", type=str, default='configs/gait_video.yaml', help="Config file")

parser.add_argument("--dataset", choices=['ucf', 'smth', 'kinetics'], default='kinetics')
parser.add_argument("--weight-path", type=str, default="*", help='Path to save weights')
parser.add_argument("--log-path", type=str, default="log", help='Path to save weights')
parser.add_argument("--resume", type=int, default=0, help='Resume training from')

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
parser.add_argument("--warmup_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=2e-4, help="Weight decay")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
# Learning scheduler
LRS = [1, 0.1, 0.01, 0.001]
STEPS = [1, 14, 25, 50]
# Parse arguments
args = parser.parse_args()
print(args)

# Load config
cfg = load_yaml(args.config)

# Load model
model = VTN(**vars(cfg))
preprocess = model.preprocess
train_preprocess = model.train_preprocess
model = model.cuda()

# Resume weights
if args.resume > 0:
    model.load_state_dict(torch.load(f'{args.weight_path}/weights_{args.resume}.pth'))

# Load dataset
if args.dataset == 'kinetics':
    train_set = Kinetics400(args.annotations, args.root_dir, preprocess=train_preprocess, frames=cfg.frames)

    val_set = Kinetics400(args.val_annotations, args.val_root_dir, preprocess=preprocess, frames=cfg.frames)
# Split
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=False)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=False)

# Tensorboard
tensorboard = SummaryWriter(args.log_path)
best_mid_acc = -1.0
best_coarse_acc = -1.0
best_avg_acc = -1.0
TREE_LOSS_WEIGHT = 0.5

# Loss and optimizer
loss_func_mid = nn.CrossEntropyLoss()
loss_func_coarse = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), momentum=0.9, lr=args.learning_rate, weight_decay=args.weight_decay)
def adjust_learning_rate(optimizer, epoch, cur_iter, max_iter):
    """Sets the learning rate to the according to POLICY"""
    for ind, step in enumerate(STEPS):
        if epoch < step:
            break
    ind = ind - 1
    lr = args.learning_rate * LRS[ind]

    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    return lr

# Hierarchical Structure Configuration
coarse_offset, num_coarse = 0, 2
mid_offset, num_mid = 2, 4
total_nodes, levels = mid_offset + num_mid, 2
mid_to_coarse_list = [0, 0, 1, 1]

fine_to_mid_list = [0, 0, 1, 2, 2, 3, 3]
# two-level hierarchy: mid -> coarse (global indices)
trees = [
    [2, 0],  # mid 0 -> coarse 0
    [3, 0],  # mid 1 -> coarse 0
    [4, 1],  # mid 2 -> coarse 1
    [5, 1]  # mid 3 -> coarse 1
]

def get_hierarchy_tensors(device):
    mid_to_coarse_tensor = torch.tensor(mid_to_coarse_list, device=device, dtype=torch.long)
    tree_loss = TreeLoss(trees, total_nodes, levels, device=device)
    return  mid_to_coarse_tensor, tree_loss

device = next(model.parameters()).device
mid_to_coarse_tensor,Tree_loss = get_hierarchy_tensors(device)
fine_to_mid_tensor = torch.tensor(fine_to_mid_list, device=device, dtype=torch.long)

for epoch in range(max(args.resume + 1, 1), args.epochs + 1):
    # Train
    model.train()
    # Adjust learning rate
    # scheduler = CosineAnnealingLR(optimizer, 100, 1e-4, -1)
    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch: {epoch}, loss: 0.000")
    mid_acc_train = 0
    coarse_acc_train = 0
    coarse_total = 0
    mid_total = 0
    for i, batch in progress:
        if len(batch) == 3:
            src, target, _ = batch
        else:
            src, target = batch
        lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        # src, target = train_loader[i]
        if torch.cuda.is_available():
            src = torch.autograd.Variable(src).cuda()
            target = torch.autograd.Variable(target).cuda()

        # Forward + backprop + optimize
        y_coarse, y_mid = model(src)
        fs_sig = torch.cat([
            torch.sigmoid(y_coarse),
            torch.sigmoid(y_mid)
        ], dim=1)
        label_id_hard = target if target.dim() == 1 else target.argmax(dim=1)

        gt_mid = fine_to_mid_tensor[label_id_hard] if label_id_hard.max().item() >= num_mid else label_id_hard
        gt_coarse = mid_to_coarse_tensor[gt_mid]
        global_leaf = (mid_offset + gt_mid).to(torch.long)
        tree_loss = Tree_loss(fs_sig, global_leaf, device=device)
        mid_logits = y_mid
        coarse_logits = y_coarse
        mid_loss = loss_func_mid(mid_logits, gt_mid)
        coarse_loss = loss_func_coarse(coarse_logits, gt_coarse)
        loss =  mid_loss + coarse_loss + tree_loss

        _, mid_pred = torch.max(y_mid, 1)
        mid_total += gt_mid.size(0)
        mid_acc_train += (mid_pred == gt_mid).cpu().sum().item()

        _, coarse_pred = torch.max(y_coarse, 1)
        coarse_total += gt_coarse.size(0)
        coarse_acc_train += (coarse_pred == gt_coarse).cpu().sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Cosine scheduler
        # scheduler.step()
        # Show loss
        loss_val = loss.item()
        progress.set_description(f"Epoch: {epoch}, loss: {loss_val}")
        # Summary
        if i % 100 == 99:
            global_step = epoch * len(train_loader) + i
            tensorboard.add_scalar('train_loss', loss_val, global_step)
            tensorboard.add_scalar('train_mid_loss', mid_loss.item(), global_step)
            tensorboard.add_scalar('train_coarse_loss', coarse_loss.item(), global_step)
            tensorboard.add_scalar('train_tree_loss', tree_loss.item(), global_step)
            tensorboard.add_scalar('lr', lr, global_step)

    print(
          "mid_acc_train:", mid_acc_train / max(1, mid_total), "coarse_acc:",
          coarse_acc_train / max(1, coarse_total))
    # Summary
    tensorboard.add_scalar('train_mid_acc', mid_acc_train / mid_total * 100, epoch)
    tensorboard.add_scalar('train_coarse_acc', coarse_acc_train / coarse_total * 100, epoch)

    # Validation
    model.eval()
    val_loss = 0.0

    mid_correct = 0
    coarse_correct = 0

    mid_total = 0
    coarse_total = 0
    Acc = Accuracy(task="multiclass", num_classes=4, average="weighted")
    Auc = AUROC(task="multiclass", num_classes=4, average='weighted')
    Spe = Specificity(task="multiclass", num_classes=4, average='weighted')
    Sen = Recall(task="multiclass", num_classes=4, average='weighted')
    F1 = F1Score(task="multiclass", num_classes=4, average='weighted')

    num_classes_mid = 4
    num_classes_coarse = 2
    # per-class probability reccoarse (dynamic by num_classes)

    data_mid = {"predict_label": [], "real_label": []}
    for i in range(num_classes_mid):
        data_mid[f"pre_class{i}"] = []

    data_coarse = {"predict_label": [], "real_label": []}
    for i in range(num_classes_coarse):
        data_coarse[f"pre_class{i}"] = []
    for batch in tqdm(val_loader, desc=f"Epoch: {epoch}, validating"):
        if len(batch) == 3:
            src, target, prefix = batch
        else:
            src, target = batch
            prefix = None
        if torch.cuda.is_available():
            src = torch.autograd.Variable(src).cuda()
            target = torch.autograd.Variable(target).cuda()

        with torch.no_grad():
            mid_to_coarse_tensor, tree_loss = get_hierarchy_tensors(src.device)
            y_coarse, y_mid = model(src)
            fs_sig = torch.cat([
                torch.sigmoid(y_coarse),
                torch.sigmoid(y_mid)
            ], dim=1)
            label_id_hard = target if target.dim() == 1 else target.argmax(dim=1)
            gt_mid = fine_to_mid_tensor[label_id_hard] if label_id_hard.max().item() >= num_mid else label_id_hard
            gt_coarse = mid_to_coarse_tensor[gt_mid]
            global_leaf = (mid_offset + gt_mid).to(torch.long)
            mid_logits = y_mid

            mid_probs = torch.softmax(y_mid, dim=1)
            preds_mid = mid_probs.argmax(dim=1)

            coarse_probs = torch.softmax(y_coarse, dim=1)
            preds_coarse = coarse_probs.argmax(dim=1)

            mid_correct += (preds_mid == gt_mid).cpu().sum().item()
            coarse_correct += (preds_coarse == gt_coarse).cpu().sum().item()

            mid_total += gt_mid.size(0)
            coarse_total += gt_coarse.size(0)

            Acc.update(preds_mid.cpu(), gt_mid.cpu())
            Auc.update(mid_probs.cpu(), gt_mid.cpu())

            Spe.update(preds_mid.cpu(), gt_mid.cpu())
            Sen.update(preds_mid.cpu(), gt_mid.cpu())
            F1.update(preds_mid.cpu(), gt_mid.cpu())

            data_mid["real_label"].extend(gt_mid.detach().cpu().tolist())
            data_mid["predict_label"].extend(preds_mid.detach().cpu().tolist())
            probs_per_class_mid = mid_probs.detach().cpu()
            for i in range(num_classes_mid):
                data_mid[f"pre_class{i}"].extend(probs_per_class_mid[:, i].tolist())
            if prefix is not None:
                if isinstance(prefix, (list, tuple)):
                    data_mid.setdefault("patient", [])
                    data_mid["patient"].extend(list(prefix))
                else:
                    data_mid.setdefault("patient", [])
                    data_mid["patient"].extend([prefix] * gt_mid.size(0))

            data_coarse["real_label"].extend(gt_coarse.detach().cpu().tolist())
            data_coarse["predict_label"].extend(preds_coarse.detach().cpu().tolist())
            probs_per_class_coarse = coarse_probs.detach().cpu()
            for i in range(num_classes_coarse):
                data_coarse[f"pre_class{i}"].extend(probs_per_class_coarse[:, i].tolist())
            if prefix is not None:
                if isinstance(prefix, (list, tuple)):
                    data_coarse.setdefault("patient", [])
                    data_coarse["patient"].extend(list(prefix))
                else:
                    data_coarse.setdefault("patient", [])
                    data_coarse["patient"].extend([prefix] * gt_mid.size(0))
    Acc_val = Acc.compute().item() * 100
    auc_val = Auc.compute().item() * 100
    spe_val = Spe.compute().item() * 100
    sen_val = Sen.compute().item() * 100
    f1_val = F1.compute().item() * 100

    data_mid = pd.DataFrame(data_mid)
    data_coarse = pd.DataFrame(data_coarse)

    mid_acc = mid_correct / max(1, mid_total)
    coarse_acc = coarse_correct / max(1, coarse_total)
    avg_acc = (mid_acc + coarse_acc) / 2.0

    print("mid_acc:", mid_acc, "coarse_acc:", coarse_acc)
    print("Val ACC:", Acc_val, "Val AUC:", auc_val,
            "Val SPE:", spe_val, "Val SEN:", sen_val,
          "Val F1:", f1_val)

    os.makedirs(args.weight_path, exist_ok=True)
    out_val_mid = os.path.join(args.weight_path, 'Val_mid.xlsx')
    out_val_coarse = os.path.join(args.weight_path, 'Val_coarse.xlsx')

    if mid_acc > best_mid_acc:
        best_mid_acc = mid_acc
        torch.save(model.state_dict(), os.path.join(args.weight_path, 'best_mid.pth'))
        data_mid.to_excel(out_val_mid, index=False)
        print(f"New best mid accuracy: {mid_acc:.4f}, saved to {out_val_mid}")
    if coarse_acc > best_coarse_acc:
        best_coarse_acc = coarse_acc
        torch.save(model.state_dict(), os.path.join(args.weight_path, 'best_coarse.pth'))
        data_coarse.to_excel(out_val_coarse, index=False)
        print(f"New best coarse accuracy: {coarse_acc:.4f}, saved to {out_val_coarse}")


    tensorboard.add_scalar('val_mid_acc', mid_acc * 100, epoch)
    tensorboard.add_scalar('val_coarse_acc', coarse_acc * 100, epoch)
    tensorboard.add_scalar('val_accuracy', Acc_val, epoch)
    tensorboard.add_scalar('val_auc', auc_val, epoch)
    tensorboard.add_scalar('val_f1', f1_val, epoch)

    # Save weights
    torch.save(model.state_dict(), f'{args.weight_path}/weights_last.pth')

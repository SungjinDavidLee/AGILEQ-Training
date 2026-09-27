def seed_everything(seed: int = 42):
    import os, random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed_everything()
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import random, numpy as np
    np.random.seed(worker_seed)
    random.seed(worker_seed)


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import argparse
import os
import sys
import logging
from datetime import datetime
import json


import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import tqdm

import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from data import Drive_Dataset, TOWNS
from configs import load_backbone_config
from models import BEVFormer

import wandb
os.environ['WANDB_API_KEY'] = '?'


def append_table_row(jsonl_path, row: dict):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_table_rows(jsonl_path):
    rows = []
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
def get_inter_union_per_sample(pred, target, threshold=0.5):

    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]

    pred_bin = (torch.sigmoid(pred) > threshold).to(torch.float32)
    tgt_bin = (target > 0.5).to(torch.float32)

    inter = (pred_bin * tgt_bin).sum(dim=(1, 2))
    union = ((pred_bin + tgt_bin) > 0).to(torch.float32).sum(dim=(1, 2))
    return inter, union


def iou_from_inter_union(inter, union):
    if isinstance(inter, torch.Tensor):
        return (inter / union.clamp(min=1e-7)).item()
    return float(inter) / max(float(union), 1e-7)


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(sum(xs) / len(xs)) if len(xs) > 0 else float("nan")

def rebuild_wandb_table_from_jsonl(jsonl_path, columns):
    table = wandb.Table(columns=columns)
    rows = load_table_rows(jsonl_path)
    for r in rows:
        img_path = r.get("img_path", "")
        if img_path and os.path.exists(img_path):
            table.add_data(
                r.get("epoch", -1),
                r.get("step", -1),
                wandb.Image(img_path, caption=r.get("caption", ""))
            )
    return table


class focalLoss(torch.nn.Module):
    def __init__(self, mode='multilabel',
                 alpha=0.75, gamma=2.0,
                 focal_weight=1.0,
                 ema_decay=0.99, eps=1e-6):
        super().__init__()
        self.focal = smp.losses.FocalLoss(mode=mode, alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.ema_decay = ema_decay
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss_focal = self.focal(y_pred, y_true)

        return loss_focal


def visualize_prediction(pred_bev_gt, pred_bev):
    return np.hstack([pred_bev, pred_bev_gt])


@torch.no_grad()
def get_inter_union(preds, targets, threshold=0.5, from_logits=True):
    if from_logits:
        preds = torch.sigmoid(preds)

    pred = preds > threshold
    tgt = targets if targets.dtype == torch.bool else (targets > 0.5)

    inter = (pred & tgt).sum().item()
    union = (pred | tgt).sum().item()
    return inter, union


def iou_from_inter_union(inter, union, eps=1e-6):
    return (inter / (union + eps)) if union > 0 else float("nan")


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


@torch.no_grad()
def get_iou(preds, targets, threshold=0.5, from_logits=True, eps=1e-6):
    inter, union = get_inter_union(preds, targets, threshold, from_logits)
    return iou_from_inter_union(inter, union, eps)


def make_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "img_val"), exist_ok=True)


def setup_logger(log_path, is_main: bool, rank: int):
    logger = logging.getLogger(f"train_logger_rank{rank}")
    logger.setLevel(logging.INFO)


    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    if is_main:
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def _to_mask2d(x, from_logits=True, threshold=0.5):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.dim() == 4:
            x = x[0, 0]
        elif x.dim() == 3:
            x = x[0]

        if from_logits:
            x = torch.sigmoid(x)

        x = (x > threshold).to(torch.uint8).cpu().numpy()
    else:
        x = (x > threshold).astype(np.uint8)

    return x


def tensor_to_bev_rgb(road, lane_solid, lane_broken, veh, ped, stop,
                      from_logits=True, draw_ego=True):
    road_m        = _to_mask2d(road,        from_logits=from_logits)
    lane_solid_m  = _to_mask2d(lane_solid,  from_logits=from_logits)
    lane_broken_m = _to_mask2d(lane_broken, from_logits=from_logits)
    veh_m         = _to_mask2d(veh,         from_logits=from_logits)
    ped_m         = _to_mask2d(ped,         from_logits=from_logits)
    stop_m        = _to_mask2d(stop,        from_logits=from_logits)

    h, w = road_m.shape
    bev = np.zeros((h, w, 3), dtype=np.uint8)


    bev[road_m == 1] = (60, 60, 60)


    bev[lane_solid_m == 1]  = (0, 255, 0)
    bev[lane_broken_m == 1] = (0, 180, 255)


    bev[stop_m == 1] = (220, 220, 220)
    bev[veh_m == 1]  = (255, 255, 0)
    bev[ped_m == 1]  = (255, 0, 255)

    if draw_ego:
        rect_w, rect_h = 12, 25
        center_x, center_y = w // 2, h // 2
        top_left = (center_x - rect_w // 2, center_y - rect_h // 2)
        bottom_right = (center_x + rect_w // 2, center_y + rect_h // 2)
        cv2.rectangle(bev, top_left, bottom_right, (255, 255, 255), thickness=-1)

    return bev

def get_model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_model_state_dict(model, state_dict):
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict)


def save_last_checkpoint(path, model, opt, sched, epoch, global_step, global_seen_samples, best_val_loss, best_val_miou, wandb_run_id):
    ckpt = {
        "model": get_model_state_dict(model),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "global_seen_samples": int(global_seen_samples),
        "best_val_loss": float(best_val_loss),
        "best_val_miou": float(best_val_miou),
        "wandb_run_id": wandb_run_id,
    }
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device)
    load_model_state_dict(model, ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    return ckpt


def ddp_setup():

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    return False, 0, 0, 1


def ddp_cleanup(is_ddp: bool):
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='BEVFormer training with W&B, resume, and DDP.')

    parser.add_argument("--v", type=int, default=2)
    parser.add_argument("--nepochs", type=int, default=100)

    parser.add_argument("--xbound", nargs=3, type=float, default=[-20.0, 20.0, 0.2])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-20.0, 20.0, 0.2])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[2.0, 50.0, 1.0])

    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--wandb_project", type=str, default="bev_final2")
    parser.add_argument("--wandb_entity", type=str, default="ropo")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--eta_min", type=float, default=1e-6)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every_pct", type=float, default=0.2)


    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    config = load_backbone_config()
    return args, config
class CombinedLoss(torch.nn.Module):
    def __init__(self, mode='multilabel',
                 alpha=0.75, gamma=2.0,
                 dice_weight=1.0, focal_weight=1.0,
                 ema_decay=0.99, eps=1e-6):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode=mode)
        self.focal = smp.losses.FocalLoss(mode=mode, alpha=alpha, gamma=gamma)

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.ema_decay = ema_decay
        self.eps = eps
        self.training=True

        self.register_buffer("ema_dice", torch.tensor(1.0))
        self.register_buffer("ema_focal", torch.tensor(1.0))

    @torch.no_grad()
    def _update_ema(self, name, value):
        buf = getattr(self, name)
        buf.mul_(self.ema_decay).add_(value * (1.0 - self.ema_decay))
    def train(self):
        self.training=True

    def val(self):
        self.training=False

    def forward(self, y_pred, y_true):
        loss_dice = self.dice(y_pred, y_true)
        loss_focal = self.focal(y_pred, y_true)

        if self.training:
            self._update_ema("ema_dice", loss_dice.detach().float())
            self._update_ema("ema_focal", loss_focal.detach().float())

        loss_dice_n = loss_dice / (self.ema_dice + self.eps)
        loss_focal_n = loss_focal / (self.ema_focal + self.eps)

        return (self.dice_weight * loss_dice_n) + (self.focal_weight * loss_focal_n)


def main():
    is_ddp, local_rank, rank, world_size = ddp_setup()
    is_main = (rank == 0)

    args, configs_img_encoder = parse_args()

    base_dir = f'run_422/bev_{args.v}'
    make_dirs(base_dir)
    logger = setup_logger(os.path.join(base_dir, "results.log"), is_main=is_main, rank=rank)
    logger.info(f"[rank={rank}/{world_size}] args: {args}")


    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    elif args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ.pop("WANDB_MODE", None)


    if not is_main:
        os.environ["WANDB_MODE"] = "disabled"

    conf = {
        'width': 800,
        'height': 448,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'config': configs_img_encoder,
    }

    wandb_config = {
        "version": args.v,
        "nepochs": args.nepochs,
        "lr": args.lr,
        "batch_size_per_gpu": args.batch_size,
        "world_size": world_size,
        "global_batch_per_step": args.batch_size * world_size,
        "accumulation_steps": args.accumulation_steps,
        "warmup_epochs": args.warmup_epochs,
        "eta_min": args.eta_min,
        "max_grad_norm": args.max_grad_norm,
        "xbound": args.xbound,
        "ybound": args.ybound,
        "zbound": args.zbound,
        "dbound": args.dbound,
        "log_every_pct": args.log_every_pct,
    }

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    last_ckpt_path = os.path.join(base_dir, "last.ckpt")
    best_loss_path = os.path.join(base_dir, "best_loss_model.pth")
    best_miou_path = os.path.join(base_dir, "best_miou_model.pth")

    train_table_jsonl = os.path.join(base_dir, "train_table.jsonl")
    val_table_jsonl = os.path.join(base_dir, "val_table.jsonl")

    resume_available = args.resume and os.path.exists(last_ckpt_path)


    prev_wandb_run_id = None
    if resume_available and is_main:
        try:
            tmp = torch.load(last_ckpt_path, map_location="cpu")
            prev_wandb_run_id = tmp.get("wandb_run_id", None)
            logger.info(f"Found last checkpoint. prev_wandb_run_id={prev_wandb_run_id}")
        except Exception as e:
            logger.info(f"Failed to read last checkpoint: {e}")
            resume_available = False

    run_name = args.run_name or f"for-v{args.v}-lr{args.lr}-bs{args.batch_size}_g{world_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run = None
    if is_main and os.environ.get("WANDB_MODE", "") != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
            id=prev_wandb_run_id,
            resume="allow" if prev_wandb_run_id else None,
            save_code=True
        )


    g = torch.Generator()
    g.manual_seed(42 + rank)


    train_set = Drive_Dataset(train=True, conf=conf,v=2)
    val_set = Drive_Dataset(train=False, conf=conf,v=2)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=8,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=8,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    model = BEVFormer(conf).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step_ratio = 5 / 20
    step_size = max(1, int(step_ratio * args.nepochs))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.8)

    loss_road = CombinedLoss(mode='multilabel', alpha=0.5, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_line = CombinedLoss(mode='multilabel', alpha=0.8, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_veh  = CombinedLoss(mode='multilabel', alpha=0.8, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_ped  = CombinedLoss(mode='multilabel', alpha=0.9, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_stop = CombinedLoss(mode='multilabel', alpha=0.7, dice_weight=0.0, focal_weight=1.0).to(device)


    if is_main and run is not None:
        val_img_table = rebuild_wandb_table_from_jsonl(
            val_table_jsonl,
            columns=["epoch", "val_step", "bev_pred_vs_gt"]
        )
        wandb.log({"val/bev_table": val_img_table}, step=0)
    else:
        val_img_table = None


    start_epoch = 0
    global_step = 0
    global_seen_samples = 0
    best_val_miou = 0.0
    best_val_loss = float("inf")

    if resume_available:
        try:
            ckpt = load_checkpoint(last_ckpt_path, model, opt, sched, device)
            start_epoch = int(ckpt["epoch"]) + 1
            global_step = int(ckpt.get("global_step", 0))
            global_seen_samples = int(ckpt.get("global_seen_samples", 0))
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            best_val_miou = float(ckpt.get("best_val_miou", best_val_miou))
            if is_main:
                logger.info(
                    f"Resumed: start_epoch={start_epoch}, global_step={global_step}, "
                    f"global_seen_samples={global_seen_samples}, best_val_loss={best_val_loss}, best_val_miou={best_val_miou}"
                )
        except Exception as e:
            if is_main:
                logger.info(f"Resume failed: {e}")


    last_idx = len(train_loader) - 1

    for epoch in range(start_epoch, args.nepochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        opt.zero_grad(set_to_none=True)


        epoch_total_samples_global = len(train_loader) * args.batch_size * world_size
        log_every_samples = max(1, int(epoch_total_samples_global * (args.log_every_pct / 100.0)))
        next_log_at = log_every_samples
        epoch_seen_samples_global = 0

        for batchi, (bev_imgs, cam_pam,
                     road_gt, lane_broken_gt,lane_solid_gt, veh_gt, ped_gt, stop_gt,town_idx, town_name_list) in enumerate(train_loader):

            road_gt = road_gt.to(device, non_blocking=True)
            lane_broken_gt = lane_broken_gt.to(device, non_blocking=True)
            lane_solid_gt = lane_solid_gt.to(device, non_blocking=True)
            veh_gt = veh_gt.to(device, non_blocking=True)
            ped_gt = ped_gt.to(device, non_blocking=True)
            stop_gt = stop_gt.to(device, non_blocking=True)

            road, lane_broken,lane_solid, veh, ped, stop = model(bev_imgs.to(device), cam_pam)


            bev_loss = 0
            bev_loss += loss_road(road, road_gt)
            bev_loss += loss_line(lane_solid, lane_solid_gt)
            bev_loss += loss_line(lane_broken, lane_broken_gt)
            bev_loss += loss_veh(veh, veh_gt)
            bev_loss += loss_ped(ped, ped_gt)
            bev_loss += loss_stop(stop, stop_gt)


            (bev_loss / args.accumulation_steps).backward()

            if (batchi + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)


            current_bs = int(bev_imgs.size(0))
            epoch_seen_samples_global += current_bs * world_size
            global_seen_samples += current_bs * world_size
            global_step += 1
            if batchi % 500 == 0:
                pred_rgb = tensor_to_bev_rgb(
                    road, lane_solid, lane_broken, veh, ped, stop,
                    from_logits=True
                )
                gt_rgb = tensor_to_bev_rgb(
                    road_gt, lane_solid_gt, lane_broken_gt, veh_gt, ped_gt, stop_gt,
                    from_logits=False
                )
                concat = visualize_prediction(gt_rgb, pred_rgb)

                save_path = os.path.join(base_dir, "img", f"debug_epoch{epoch}_valb{int(batchi)}.png")
                cv2.imwrite(save_path, concat)

            should_log = False
            if epoch_seen_samples_global >= next_log_at:
                should_log = True
                while epoch_seen_samples_global >= next_log_at:
                    next_log_at += log_every_samples

            if should_log and is_main:
                DAS_iou         = get_iou(road, road_gt)
                LaneSolid_iou   = get_iou(lane_solid, lane_solid_gt)
                LaneBroken_iou  = get_iou(lane_broken, lane_broken_gt)
                Vehicle_iou     = get_iou(veh, veh_gt)
                Walker_iou      = get_iou(ped, ped_gt)
                Stop_iou        = get_iou(stop, stop_gt)

                m_iou_6cls = nanmean([
                    DAS_iou,
                    LaneSolid_iou,
                    LaneBroken_iou,
                    Vehicle_iou,
                    Walker_iou,
                    Stop_iou,
                ])

                logger.info(
                    f"TRAIN[{epoch}] [{batchi:>4d}/{last_idx}] "
                    f"Loss: {bev_loss.item():>7.4f} "
                    f"veh_iou {Vehicle_iou:>7.3f} "
                    f"ped_iou {Walker_iou:>7.3f} "
                    f"lane_solid_iou {LaneSolid_iou:>7.3f} "
                    f"lane_broken_iou {LaneBroken_iou:>7.3f} "
                    f"road_iou {DAS_iou:>7.3f} "
                    f"stop_iou {Stop_iou:>7.3f} "
                    f"miou_6cls {m_iou_6cls:>7.3f}"
                )

                if run is not None:
                    wandb.log({
                        "train/loss": bev_loss.item(),
                        "train/iou_vehicle": Vehicle_iou,
                        "train/iou_ped": Walker_iou,
                        "train/iou_lane_solid": LaneSolid_iou,
                        "train/iou_lane_broken": LaneBroken_iou,
                        "train/iou_road": DAS_iou,
                        "train/iou_stop": Stop_iou,
                        "train/miou_6cls": m_iou_6cls,
                        "lr": opt.param_groups[0]["lr"],
                        "epoch": epoch,
                        "global_step": global_step,
                        "epoch_progress": float(epoch_seen_samples_global) / float(max(1, epoch_total_samples_global))
                    }, step=int(global_seen_samples))

        if (len(train_loader) % args.accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)


        if is_main:
            logger.info("VAL")
        model.eval()

        NUM_TOWNS = len(TOWNS)
        val_loss_sum = 0.0

        road_I = road_U = 0.0
        lane_solid_I = lane_solid_U = 0.0
        lane_broken_I = lane_broken_U = 0.0
        veh_I = veh_U = 0.0
        ped_I = ped_U = 0.0
        stop_I = stop_U = 0.0

        num_val_batches = 0.0

        town_loss_sum = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_num_samples = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)

        town_road_I = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_road_U = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_lane_solid_I = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_lane_solid_U = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_lane_broken_I = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_lane_broken_U = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)

        town_veh_I  = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_veh_U  = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_ped_I  = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_ped_U  = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_stop_I = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)
        town_stop_U = torch.zeros(NUM_TOWNS, dtype=torch.float64, device=device)

        with torch.no_grad():
            for vi, (
                bev_imgs, cam_pam,
                road_gt, lane_broken_gt, lane_solid_gt, veh_gt, ped_gt, stop_gt,
                town_idx, town_name_list
            ) in tqdm.tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                disable=not is_main
            ):
                road_gt         = road_gt.to(device, non_blocking=True)
                lane_broken_gt  = lane_broken_gt.to(device, non_blocking=True)
                lane_solid_gt   = lane_solid_gt.to(device, non_blocking=True)
                veh_gt          = veh_gt.to(device, non_blocking=True)
                ped_gt          = ped_gt.to(device, non_blocking=True)
                stop_gt         = stop_gt.to(device, non_blocking=True)
                town_idx        = town_idx.to(device, non_blocking=True)

                road, lane_broken, lane_solid, veh, ped, stop = model(bev_imgs.to(device), cam_pam)
                bev_loss = 0
                bev_loss += loss_road(road, road_gt)
                bev_loss += loss_line(lane_solid, lane_solid_gt)
                bev_loss += loss_line(lane_broken, lane_broken_gt)
                bev_loss += loss_veh(veh, veh_gt)
                bev_loss += loss_ped(ped, ped_gt)
                bev_loss += loss_stop(stop, stop_gt)

                val_loss_sum += bev_loss.item()
                i, u = get_inter_union(road, road_gt)
                road_I += i; road_U += u

                i, u = get_inter_union(lane_solid, lane_solid_gt)
                lane_solid_I += i; lane_solid_U += u

                i, u = get_inter_union(lane_broken, lane_broken_gt)
                lane_broken_I += i; lane_broken_U += u

                i, u = get_inter_union(veh, veh_gt)
                veh_I += i; veh_U += u

                i, u = get_inter_union(ped, ped_gt)
                ped_I += i; ped_U += u

                i, u = get_inter_union(stop, stop_gt)
                stop_I += i; stop_U += u

                num_val_batches += 1.0


                road_i_b, road_u_b = get_inter_union_per_sample(road, road_gt)
                lane_solid_i_b, lane_solid_u_b = get_inter_union_per_sample(lane_solid, lane_solid_gt)
                lane_broken_i_b, lane_broken_u_b = get_inter_union_per_sample(lane_broken, lane_broken_gt)
                veh_i_b, veh_u_b = get_inter_union_per_sample(veh, veh_gt)
                ped_i_b, ped_u_b = get_inter_union_per_sample(ped, ped_gt)
                stop_i_b, stop_u_b = get_inter_union_per_sample(stop, stop_gt)

                bs = town_idx.shape[0]
                per_sample_loss = bev_loss.detach().to(torch.float64) / max(bs, 1)

                for b in range(bs):
                    t = int(town_idx[b].item())

                    town_loss_sum[t] += per_sample_loss
                    town_num_samples[t] += 1.0

                    town_road_I[t] += road_i_b[b].double()
                    town_road_U[t] += road_u_b[b].double()
                    town_lane_solid_I[t] += lane_solid_i_b[b].double()
                    town_lane_solid_U[t] += lane_solid_u_b[b].double()

                    town_lane_broken_I[t] += lane_broken_i_b[b].double()
                    town_lane_broken_U[t] += lane_broken_u_b[b].double()

                    town_veh_I[t] += veh_i_b[b].double()
                    town_veh_U[t] += veh_u_b[b].double()

                    town_ped_I[t] += ped_i_b[b].double()
                    town_ped_U[t] += ped_u_b[b].double()

                    town_stop_I[t] += stop_i_b[b].double()
                    town_stop_U[t] += stop_u_b[b].double()


                if is_main and (int(num_val_batches) % 25 == 0):
                    pred_rgb = tensor_to_bev_rgb(
                        road, lane_solid, lane_broken, veh, ped, stop,
                        from_logits=True
                    )
                    gt_rgb = tensor_to_bev_rgb(
                        road_gt, lane_solid_gt, lane_broken_gt, veh_gt, ped_gt, stop_gt,
                        from_logits=False
                    )
                    concat = visualize_prediction(gt_rgb, pred_rgb)

                    save_path = os.path.join(base_dir, "img_val", f"debug_epoch{epoch}_valb{int(num_val_batches)}.png")
                    cv2.imwrite(save_path, concat)

                    append_table_row(val_table_jsonl, {
                        "epoch": epoch,
                        "step": int(num_val_batches),
                        "img_path": save_path,
                        "caption": f"val e{epoch} b{int(num_val_batches)}"
                    })

                    if run is not None and val_img_table is not None:
                        val_img_table.add_data(
                            epoch,
                            int(num_val_batches),
                            wandb.Image(save_path, caption=f"val e{epoch} b{int(num_val_batches)}")
                        )
                        wandb.log({"val/bev_table": val_img_table}, step=int(global_seen_samples))


        if is_ddp:
            t = torch.tensor([
                val_loss_sum,
                road_I, road_U,
                lane_solid_I, lane_solid_U,
                lane_broken_I, lane_broken_U,
                veh_I, veh_U,
                ped_I, ped_U,
                stop_I, stop_U,
                num_val_batches
            ], device=device, dtype=torch.float64)

            dist.all_reduce(t, op=dist.ReduceOp.SUM)

            (
                val_loss_sum,
                road_I, road_U,
                lane_solid_I, lane_solid_U,
                lane_broken_I, lane_broken_U,
                veh_I, veh_U,
                ped_I, ped_U,
                stop_I, stop_U,
                num_val_batches
            ) = t.tolist()

        avg_val_loss = val_loss_sum / max(1.0, num_val_batches)

        avg_val_road_iou        = iou_from_inter_union(road_I, road_U)
        avg_val_lane_solid_iou  = iou_from_inter_union(lane_solid_I, lane_solid_U)
        avg_val_lane_broken_iou = iou_from_inter_union(lane_broken_I, lane_broken_U)
        avg_val_vehicle_iou     = iou_from_inter_union(veh_I, veh_U)
        avg_val_walker_iou      = iou_from_inter_union(ped_I, ped_U)
        avg_val_stop_iou        = iou_from_inter_union(stop_I, stop_U)

        avg_val_miou_6cls = nanmean([
            avg_val_road_iou,
            avg_val_lane_solid_iou,
            avg_val_lane_broken_iou,
            avg_val_vehicle_iou,
            avg_val_walker_iou,
            avg_val_stop_iou,
        ])

        if is_main:
            logger.info(
                f"VAL[{epoch}] Loss: {avg_val_loss:>7.4f} "
                f"vehicle_iou: {avg_val_vehicle_iou:>7.3f} "
                f"walker_iou: {avg_val_walker_iou:>7.3f} "
                f"lane_solid_iou: {avg_val_lane_solid_iou:>7.3f} "
                f"lane_broken_iou: {avg_val_lane_broken_iou:>7.3f} "
                f"road_iou: {avg_val_road_iou:>7.3f} "
                f"stop_iou: {avg_val_stop_iou:>7.3f} "
                f"miou_6cls: {avg_val_miou_6cls:>7.3f}"
            )

            log_dict = {
                "val/loss": avg_val_loss,
                "val/iou_vehicle": avg_val_vehicle_iou,
                "val/iou_ped": avg_val_walker_iou,
                "val/iou_lane_solid": avg_val_lane_solid_iou,
                "val/iou_lane_broken": avg_val_lane_broken_iou,
                "val/iou_road": avg_val_road_iou,
                "val/iou_stop": avg_val_stop_iou,
                "val/miou_6cls": avg_val_miou_6cls,
                "epoch": epoch,
            }


            for ti, town_name in enumerate(TOWNS):
                n = float(town_num_samples[ti].item())

                if n <= 0:
                    continue

                town_avg_loss = float(town_loss_sum[ti].item() / n)

                town_avg_road_iou = iou_from_inter_union(
                    float(town_road_I[ti].item()), float(town_road_U[ti].item())
                )
                town_avg_lane_solid_iou = iou_from_inter_union(
                    float(town_lane_solid_I[ti].item()), float(town_lane_solid_U[ti].item())
                )
                town_avg_lane_broken_iou = iou_from_inter_union(
                    float(town_lane_broken_I[ti].item()), float(town_lane_broken_U[ti].item())
                )

                town_avg_vehicle_iou = iou_from_inter_union(
                    float(town_veh_I[ti].item()), float(town_veh_U[ti].item())
                )
                town_avg_walker_iou = iou_from_inter_union(
                    float(town_ped_I[ti].item()), float(town_ped_U[ti].item())
                )
                town_avg_stop_iou = iou_from_inter_union(
                    float(town_stop_I[ti].item()), float(town_stop_U[ti].item())
                )
                town_avg_miou = nanmean([
                    town_avg_road_iou,
                    town_avg_lane_solid_iou,
                    town_avg_lane_broken_iou,
                    town_avg_vehicle_iou,
                    town_avg_walker_iou,
                    town_avg_stop_iou,
                ])

                logger.info(
                    f"VAL[{epoch}][{town_name}] "
                    f"Loss: {town_avg_loss:>7.4f} "
                    f"vehicle_iou: {town_avg_vehicle_iou:>7.3f} "
                    f"walker_iou: {town_avg_walker_iou:>7.3f} "
                    f"lane_solid_iou: {town_avg_lane_solid_iou:>7.3f} "
                    f"lane_broken_iou: {town_avg_lane_broken_iou:>7.3f} "
                    f"road_iou: {town_avg_road_iou:>7.3f} "
                    f"stop_iou: {town_avg_stop_iou:>7.3f} "
                    f"miou: {town_avg_miou:>7.3f}"
                )


                prefix = f"val_{town_name}"
                log_dict[f"{prefix}/loss"] = town_avg_loss
                log_dict[f"{prefix}/iou_vehicle"] = town_avg_vehicle_iou
                log_dict[f"{prefix}/iou_ped"] = town_avg_walker_iou
                log_dict[f"{prefix}/iou_lane_solid"] = town_avg_lane_solid_iou
                log_dict[f"{prefix}/iou_lane_broken"] = town_avg_lane_broken_iou
                log_dict[f"{prefix}/iou_road"] = town_avg_road_iou
                log_dict[f"{prefix}/iou_stop"] = town_avg_stop_iou
                log_dict[f"{prefix}/miou"] = town_avg_miou
                log_dict[f"{prefix}/num_samples"] = n

            if run is not None:
                wandb.log(log_dict, step=int(global_seen_samples))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(get_model_state_dict(model), best_loss_path)

                if run is not None:
                    art = wandb.Artifact(
                        name=f"{wandb.run.name}-best-loss",
                        type="model",
                        metadata={"best_val_loss": best_val_loss, "epoch": epoch, "global_seen_samples": int(global_seen_samples)}
                    )
                    art.add_file(best_loss_path)
                    wandb.log_artifact(art)

            if avg_val_miou_6cls > best_val_miou:
                best_val_miou = avg_val_miou_6cls
                torch.save(get_model_state_dict(model), best_miou_path)

                if run is not None:
                    art = wandb.Artifact(
                        name=f"{wandb.run.name}-best-miou",
                        type="model",
                        metadata={"best_val_miou": best_val_miou, "epoch": epoch, "global_seen_samples": int(global_seen_samples)}
                    )
                    art.add_file(best_miou_path)
                    wandb.log_artifact(art)


            save_last_checkpoint(
                last_ckpt_path,
                model, opt, sched,
                epoch=epoch,
                global_step=global_step,
                global_seen_samples=int(global_seen_samples),
                best_val_loss=best_val_loss,
                best_val_miou=best_val_miou,
                wandb_run_id=(wandb.run.id if run is not None else None)
            )


        sched.step()

    if is_main and run is not None:
        wandb.finish()

    ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()

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


    torch.use_deterministic_algorithms(True)


    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed_everything()
def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    import random, numpy as np
    np.random.seed(worker_seed)
    random.seed(worker_seed)

import argparse
import os
import sys
import logging
from datetime import datetime
import json

sys.path.append('.')

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import tqdm

import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from data import Drive_Dataset
from configs.config import get_config_b, get_config_t
from models_carla.bevformer import bevformer

import wandb
os.environ['WANDB_API_KEY'] = '???'


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


def rebuild_wandb_table_from_jsonl(jsonl_path, columns):
    table = wandb.Table(columns=columns, log_mode="MUTABLE")
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


def visualize_prediction(pred_bev_gt, pred_bev):
    return np.hstack([pred_bev, pred_bev_gt])


@torch.no_grad()
def get_inter_union(preds, targets, threshold=0.5, from_logits=True):
    if from_logits:
        preds = torch.sigmoid(preds)

    pred = preds > threshold
    tgt  = targets if targets.dtype == torch.bool else (targets > 0.5)

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


def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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


def tensor_to_bev_rgb(road, lane, veh, ped, stop):
    b, c, h, w = road.shape
    pred_bev = np.zeros((h, w, 3), dtype=np.uint8)

    lane = torch.sigmoid(lane).detach().cpu().numpy() * 255
    veh = torch.sigmoid(veh).detach().cpu().numpy() * 255
    ped = torch.sigmoid(ped).detach().cpu().numpy() * 255
    road = torch.sigmoid(road).detach().cpu().numpy() * 255
    stop = torch.sigmoid(stop).detach().cpu().numpy() * 255

    pred_bev[road[0][0] > 128] = (60, 60, 60)
    pred_bev[lane[0][0] > 128] = (0, 100, 0)
    pred_bev[veh[0][0] > 128] = (255, 255, 0)
    pred_bev[ped[0][0] > 128] = (255, 0, 255)
    pred_bev[stop[0][0] > 128] = (200, 200, 200)

    rect_w, rect_h = 12, 25
    center_x, center_y = w // 2, h // 2
    top_left = (center_x - rect_w // 2, center_y - rect_h // 2)
    bottom_right = (center_x + rect_w // 2, center_y + rect_h // 2)
    cv2.rectangle(pred_bev, top_left, bottom_right, (255, 255, 255), thickness=-1)

    return pred_bev


def save_last_checkpoint(path, model, opt, sched, epoch, global_step, global_seen_samples, best_val_loss, best_val_miou, wandb_run_id):
    ckpt = {
        "model": model.state_dict(),
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
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    return ckpt


def parse_args():
    parser = argparse.ArgumentParser(description='Vision2Drive train w/ W&B + resume.')

    parser.add_argument("--v", type=int, default=553)
    parser.add_argument("--nepochs", type=int, default=100)

    parser.add_argument("--xbound", nargs=3, type=float, default=[-20.0, 20.0, 0.2])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-20.0, 20.0, 0.2])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])

    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--wandb_project", type=str, default="bev_final")
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

    args = parser.parse_args()
    config = get_config_t()
    return args, config


def main():
    args, configs_img_encoder = parse_args()

    base_dir = f'run/bev_{args.v}'
    make_dirs(base_dir)
    logger = setup_logger(os.path.join(base_dir, "results.log"))
    logger.info("args: %s", args)

    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    elif args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ.pop("WANDB_MODE", None)

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
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "warmup_epochs": args.warmup_epochs,
        "eta_min": args.eta_min,
        "max_grad_norm": args.max_grad_norm,
        "xbound": args.xbound,
        "ybound": args.ybound,
        "zbound": args.zbound,
        "log_every_pct": args.log_every_pct,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_ckpt_path = os.path.join(base_dir, "last.ckpt")
    best_loss_path = os.path.join(base_dir, "best_loss_model.pth")
    best_miou_path = os.path.join(base_dir, "best_miou_model.pth")

    train_table_jsonl = os.path.join(base_dir, "train_table.jsonl")
    val_table_jsonl = os.path.join(base_dir, "val_table.jsonl")

    resume_available = args.resume and os.path.exists(last_ckpt_path)
    prev_wandb_run_id = None
    if resume_available:
        try:
            tmp = torch.load(last_ckpt_path, map_location="cpu")
            prev_wandb_run_id = tmp.get("wandb_run_id", None)
            logger.info(f"Found last checkpoint. prev_wandb_run_id={prev_wandb_run_id}")
        except Exception as e:
            logger.info(f"Failed to read last checkpoint: {e}")
            resume_available = False

    run_name = args.run_name or f"bev-v{args.v}-lr{args.lr}-bs{args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    g.manual_seed(42)

    train_set = Drive_Dataset(train=True, conf=conf)
    val_set = Drive_Dataset(train=False, conf=conf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,
                            )

    model = bevformer(conf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)


    step_ratio = 5 / 20
    step_size = max(1, int(step_ratio * args.nepochs))


    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.8)
    loss_road = CombinedLoss(mode='multilabel', alpha=0.5, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_line = CombinedLoss(mode='multilabel', alpha=0.8, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_veh  = CombinedLoss(mode='multilabel', alpha=0.8, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_ped  = CombinedLoss(mode='multilabel', alpha=0.9, dice_weight=0.0, focal_weight=1.0).to(device)
    loss_stop = CombinedLoss(mode='multilabel', alpha=0.7, dice_weight=0.0, focal_weight=1.0).to(device)


    val_img_table = rebuild_wandb_table_from_jsonl(
        val_table_jsonl,
        columns=["epoch", "val_step", "bev_pred_vs_gt"]
    )

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
            logger.info(f"Resumed: start_epoch={start_epoch}, global_step={global_step}, global_seen_samples={global_seen_samples}, best_val_loss={best_val_loss}, best_val_miou={best_val_miou}")
        except Exception as e:
            logger.info(f"Resume failed: {e}")


    wandb.log({"val/bev_table": val_img_table}, step=global_seen_samples)

    last_idx = len(train_loader) - 1
    for epoch in range(start_epoch, args.nepochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_road.train()
        loss_line.train()
        loss_stop.train()
        loss_veh.train()
        loss_ped.train()
        epoch_total_samples = len(train_loader) * args.batch_size
        log_every_samples = max(1, int(epoch_total_samples * (args.log_every_pct / 100.0)))
        next_log_at = log_every_samples
        epoch_seen_samples = 0

        for batchi, (bev_imgs, topview_img, cam_pam, lidar_data, lidar_mask,
                     road_gt, lane_gt, veh_gt, ped_gt, stop_gt, route_gt) in enumerate(train_loader):

            road_gt = road_gt.to(device)
            lane_gt = lane_gt.to(device)
            veh_gt = veh_gt.to(device)
            ped_gt = ped_gt.to(device)
            stop_gt = stop_gt.to(device)
            route_gt = route_gt.to(device)

            road, lane, veh, ped, stop = model(bev_imgs.to(device), cam_pam)

            bev_loss = 0
            bev_loss += loss_road(road, road_gt)
            bev_loss += loss_line(lane, lane_gt)
            bev_loss += loss_veh(veh, veh_gt)
            bev_loss += loss_ped(ped, ped_gt)
            bev_loss += loss_stop(stop, stop_gt)

            (bev_loss / args.accumulation_steps).backward()

            if (batchi + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            current_bs = int(bev_imgs.size(0))
            epoch_seen_samples += current_bs
            global_seen_samples += current_bs
            global_step += 1

            should_log = False
            if epoch_seen_samples >= next_log_at:
                should_log = True
                while epoch_seen_samples >= next_log_at:
                    next_log_at += log_every_samples

            if should_log:
                DAS_iou     = get_iou(road, road_gt)
                Lane_iou    = get_iou(lane, lane_gt)
                Vehicle_iou = get_iou(veh,  veh_gt)
                Walker_iou  = get_iou(ped,  ped_gt)
                Stop_iou    = get_iou(stop, stop_gt)

                m_iou = nanmean([DAS_iou, Lane_iou, Vehicle_iou, Walker_iou, Stop_iou])
                logger.info(
                    f"TRAIN[{epoch}] [{batchi:>4d}/{last_idx}] "
                    f"Loss: {bev_loss.item():>7.4f} "
                    f"veh_iou{Vehicle_iou:>7.3f} "
                    f"ped_iou{Walker_iou:>7.3f} "
                    f"Lane_iou{Lane_iou:>7.3f} "
                    f"DAS_iou{DAS_iou:>7.3f} "
                    f"stop_iou{Stop_iou:>7.3f} "
                    f"m_iou{m_iou:>7.3f}"
                )

                wandb.log({
                    "train/loss": bev_loss.item(),
                    "train/iou_vehicle": Vehicle_iou,
                    "train/iou_ped": Walker_iou,
                    "train/iou_lane": Lane_iou,
                    "train/iou_road": DAS_iou,
                    "train/iou_stop": Stop_iou,
                    "train/miou": m_iou,
                    "lr": opt.param_groups[0]["lr"],
                    "epoch": epoch,
                    "global_step": global_step,
                    "epoch_progress": float(epoch_seen_samples) / float(max(1, epoch_total_samples))
                }, step=int(global_seen_samples))


        if (len(train_loader) % args.accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)

        logger.info("VAL")
        model.eval()

        loss_road.val()
        loss_line.val()
        loss_stop.val()
        loss_veh.val()
        loss_ped.val()
        num_val_batches = 0
        val_loss_sum = 0.0

        road_I = road_U = 0.0
        lane_I = lane_U = 0.0
        veh_I  = veh_U  = 0.0
        ped_I  = ped_U  = 0.0
        stop_I = stop_U = 0.0

        with torch.no_grad():
            for vi, (bev_imgs, topview_img, cam_pam, lidar_data, lidar_mask,
                     road_gt, lane_gt, veh_gt, ped_gt, stop_gt, route_gt) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):

                road_gt = road_gt.to(device)
                lane_gt = lane_gt.to(device)
                veh_gt = veh_gt.to(device)
                ped_gt = ped_gt.to(device)
                stop_gt = stop_gt.to(device)
                route_gt = route_gt.to(device)

                road, lane, veh, ped, stop = model(bev_imgs.to(device), cam_pam)

                bev_loss = 0
                bev_loss += loss_road(road, road_gt)
                bev_loss += loss_line(lane, lane_gt)
                bev_loss += loss_veh(veh, veh_gt)
                bev_loss += loss_ped(ped, ped_gt)
                bev_loss += loss_line(stop, stop_gt)


                val_loss_sum += bev_loss.item()

                i, u =  get_inter_union(road, road_gt);   road_I += i;  road_U += u
                i, u =  get_inter_union(lane, lane_gt);   lane_I += i;  lane_U += u
                i, u = get_inter_union(veh,  veh_gt);    veh_I  += i;  veh_U  += u
                i, u = get_inter_union(ped,  ped_gt);    ped_I  += i;  ped_U  += u
                i, u = get_inter_union(stop, stop_gt);   stop_I += i;  stop_U += u

                num_val_batches += 1

                if num_val_batches % 1000 == 0:
                    pred_rgb = tensor_to_bev_rgb(road, lane, veh, ped, stop)
                    gt_rgb = tensor_to_bev_rgb(road_gt, lane_gt, veh_gt, ped_gt, stop_gt)
                    concat = visualize_prediction(gt_rgb, pred_rgb)

                    save_path = os.path.join(base_dir, "img_val", f"debug_epoch{epoch}_valb{num_val_batches}.png")
                    cv2.imwrite(save_path, concat)

                    append_table_row(val_table_jsonl, {
                        "epoch": epoch,
                        "step": num_val_batches,
                        "img_path": save_path,
                        "caption": f"val e{epoch} b{num_val_batches}"
                    })

                    val_img_table.add_data(epoch, num_val_batches, wandb.Image(save_path, caption=f"val e{epoch} b{num_val_batches}"))
                    wandb.log({"val/bev_table": val_img_table}, step=int(global_seen_samples))

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        avg_val_road_iou    = iou_from_inter_union(road_I, road_U)
        avg_val_lane_iou    = iou_from_inter_union(lane_I, lane_U)
        avg_val_vehicle_iou = iou_from_inter_union(veh_I,  veh_U)
        avg_val_walker_iou  = iou_from_inter_union(ped_I,  ped_U)
        avg_val_stop_iou    = iou_from_inter_union(stop_I, stop_U)

        avg_val_miou = nanmean([avg_val_road_iou, avg_val_lane_iou, avg_val_vehicle_iou,
                        avg_val_walker_iou, avg_val_stop_iou])

        logger.info(
            f"VAL[{epoch}] Loss: {avg_val_loss:>7.4f} "
            f"vehicle_iou: {avg_val_vehicle_iou:>7.3f} "
            f"walker_iou: {avg_val_walker_iou:>7.3f} "
            f"Lane_iou: {avg_val_lane_iou:>7.3f} "
            f"Road_iou: {avg_val_road_iou:>7.3f} "
            f"stop_iou: {avg_val_stop_iou:>7.3f} "
            f"m_iou: {avg_val_miou:>7.3f}"
        )

        wandb.log({
            "val/loss": avg_val_loss,
            "val/iou_vehicle": avg_val_vehicle_iou,
            "val/iou_ped": avg_val_walker_iou,
            "val/iou_lane": avg_val_lane_iou,
            "val/iou_road": avg_val_road_iou,
            "val/iou_stop": avg_val_stop_iou,
            "val/miou": avg_val_miou,
            "epoch": epoch
        }, step=int(global_seen_samples))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_loss_path)
            art = wandb.Artifact(
                name=f"{wandb.run.name}-best-loss",
                type="model",
                metadata={"best_val_loss": best_val_loss, "epoch": epoch, "global_seen_samples": int(global_seen_samples)}
            )
            art.add_file(best_loss_path)
            wandb.log_artifact(art)

        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            torch.save(model.state_dict(), best_miou_path)
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
            wandb_run_id=wandb.run.id
        )

        sched.step()

    wandb.finish()


if __name__ == "__main__":
    main()

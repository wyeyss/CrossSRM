"""Train the CrossSRM publication model.

The fixed architecture uses multi-view and support-conditioned graphs,
fixed graph mixing, last-value residuals, and adapter/head/spatial-branch
target adaptation. Target support/test preprocessing is unchanged.

Example: python train.py --seed 7"""
from __future__ import annotations
import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import traffic_dataset
from utils import get_data_list, set_seed
from model import CrossSRM


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def log(msg: str):
    print(f"[{now()}] {msg}")


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_batches(
    num_samples: int, batch_size: int, shuffle: bool, seed: int, drop_last: bool = True
) -> List[np.ndarray]:
    idx = np.arange(num_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    batches: List[np.ndarray] = []
    for st in range(0, len(idx), batch_size):
        sub = idx[st : st + batch_size]
        if len(sub) == batch_size or not drop_last:
            batches.append(sub)
    return batches


def get_city_tensors(ds: traffic_dataset, city: str):
    x = ds.x_list[city]
    y = ds.y_list[city]
    A = ds.A_list[city]
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    if not torch.is_tensor(y):
        y = torch.tensor(y).float()
    if not torch.is_tensor(A):
        A = torch.tensor(A).float()
    return (x.float(), y.float(), A.float())


def make_dataset_norm_stats(
    ds: traffic_dataset, city: str, y: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    mean = float(ds.means_list[city][0])
    std = max(float(ds.stds_list[city][0]), 1e-06)
    out = {"mean": mean, "std": std, "name": city, "source": "dataset_input_norm"}
    if y is not None:
        yy = y.detach().float().reshape(-1)
        out["min"] = float(yy.min().item())
        out["max"] = float(yy.max().item())
    return out


def norm_y(y: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    return (y - float(stats["mean"])) / (float(stats["std"]) + 1e-06)


def denorm_y(y_norm: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    return y_norm * (float(stats["std"]) + 1e-06) + float(stats["mean"])


def prepare_x(x: torch.Tensor) -> torch.Tensor:
    """Final data convention: only keep x[..., 0:1]. No extra input normalization."""
    return x[..., :1].contiguous()


def cut_input(x: torch.Tensor, input_len: int) -> torch.Tensor:
    if input_len <= 0:
        return x
    return x[:, :, -int(input_len) :, :].contiguous()


def metric_np(pred: np.ndarray, y: np.ndarray, eps: float = 1e-05) -> Dict[str, List[float]]:
    out = {"MSE": [], "RMSE": [], "MAE": [], "MAPE": []}
    for h in range(pred.shape[-1]):
        p = pred[:, :, h].reshape(-1)
        gt = y[:, :, h].reshape(-1)
        mse = float(np.mean((p - gt) ** 2))
        out["MSE"].append(mse)
        out["RMSE"].append(float(mse**0.5))
        out["MAE"].append(float(np.mean(np.abs(p - gt))))
        out["MAPE"].append(float(np.mean(np.abs(p - gt) / np.maximum(np.abs(gt), eps))))
    return out


def average_metrics(m: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in m.items()}


def print_final_metrics(m: Dict[str, List[float]], title: str = "FINAL-TEST") -> None:
    avg = average_metrics(m)
    log(f"========== {title} ==========")
    for h in range(len(m["MAE"])):
        log(
            f"H{h + 1:02d}: MSE={m['MSE'][h]:.6f} RMSE={m['RMSE'][h]:.6f} MAE={m['MAE'][h]:.6f} MAPE={m['MAPE'][h]:.6f}"
        )
    log(
        f"Average: MSE={avg['MSE']:.6f} RMSE={avg['RMSE']:.6f} MAE={avg['MAE']:.6f} MAPE={avg['MAPE']:.6f}"
    )


def split_target_support_indices(
    num_windows: int,
    his_num: int,
    pred_num: int,
    target_days: int,
    support_train_days: int,
    stride: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, val_idx = ([], [])
    train_end = int(support_train_days * 288)
    support_end = int(target_days * 288)
    for i in range(num_windows):
        x_start = i * stride
        y_start = x_start + his_num
        y_end = y_start + pred_num
        if y_end <= train_end:
            train_idx.append(i)
        elif y_start >= train_end and y_end <= support_end:
            val_idx.append(i)
    return (np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64))


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def collect_scalar_diag(aux: Dict[str, torch.Tensor], store: Dict[str, List[float]]) -> None:
    for k, v in aux.items():
        if torch.is_tensor(v) and v.numel() == 1:
            store.setdefault(k, []).append(float(v.detach().cpu().item()))


def set_finetune_params(model: CrossSRM, keys: List[str]) -> Tuple[int, int]:
    total = sum((p.numel() for p in model.parameters()))
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any((k in name for k in keys)):
            p.requires_grad = True
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (total, trainable)


@torch.no_grad()
def build_task_graph_from_support(
    model: CrossSRM, support_x: torch.Tensor, A: torch.Tensor, args, device: torch.device
):
    sx = prepare_x(cut_input(support_x.to(device), args.input_len))
    A_dev = A.to(device)
    return model.build_task_graph(sx, A_dev)


@torch.no_grad()
def evaluate_city(
    model: CrossSRM,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    input_len: int,
    use_target_adapter: bool,
    y_stats: Dict[str, float],
    A: Optional[torch.Tensor] = None,
    A_task: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, List[float]], np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    preds, ys, losses = ([], [], [])
    diag_store: Dict[str, List[float]] = {}
    batches = make_batches(x.shape[0], batch_size, shuffle=False, seed=0, drop_last=False)
    for idx in batches:
        xb = prepare_x(cut_input(x[idx].to(device), input_len))
        yb_raw = y[idx].to(device)
        A_dev = A.to(device) if A is not None else None
        pred_norm, aux = model(
            xb, use_target_adapter=use_target_adapter, A_static=A_dev, A_task=A_task
        )
        collect_scalar_diag(aux, diag_store)
        pred_raw = denorm_y(pred_norm, y_stats)
        losses.append(float(F.l1_loss(pred_raw, yb_raw).detach().cpu().item()))
        preds.append(pred_raw.detach().cpu().numpy())
        ys.append(yb_raw.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    y_np = np.concatenate(ys, axis=0)
    metrics = metric_np(pred_np, y_np)
    diag = {k: float(np.mean(v)) for k, v in diag_store.items() if v}
    return (float(np.mean(losses)), metrics, pred_np, y_np, diag)


def source_city_val_loss(
    model, source_data, source_stats, source_cities, args, device
) -> Tuple[float, Dict[str, float]]:
    vals, details = ([], {})
    for city in source_cities:
        x, y, A = source_data[city]
        split = max(1, int(x.shape[0] * args.source_train_ratio))
        xv, yv = (x[split:], y[split:])
        if xv.shape[0] < args.batch_size:
            xv, yv = (x[-args.batch_size :], y[-args.batch_size :])
        A_task, _ = build_task_graph_from_support(
            model, x[:split][: max(1, min(args.task_support_windows, split))], A, args, device
        )
        loss, _, _, _, _ = evaluate_city(
            model,
            xv,
            yv,
            args.batch_size,
            device,
            input_len=args.input_len,
            use_target_adapter=False,
            y_stats=source_stats[city],
            A=A,
            A_task=A_task,
        )
        vals.append(loss)
        details[city] = loss
    return (float(np.mean(vals)), details)


def train_source(
    model, source_data, source_stats, source_cities, args, device, save_dir: Path
) -> Path:
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.source_lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_val = float("inf")
    best_path = save_dir / "pretrain_best.pt"
    for epoch in range(1, args.source_epochs + 1):
        model.train()
        city_batches: Dict[str, List[np.ndarray]] = {}
        min_steps = None
        for ci, city in enumerate(source_cities):
            x, _, _ = source_data[city]
            split = max(args.batch_size, int(x.shape[0] * args.source_train_ratio))
            batches = make_batches(
                split, args.batch_size, shuffle=True, seed=args.seed + epoch * 100 + ci
            )
            city_batches[city] = batches
            min_steps = len(batches) if min_steps is None else min(min_steps, len(batches))
        if not min_steps:
            raise RuntimeError("No source training batches.")
        epoch_pred_losses = []
        diag_store: Dict[str, List[float]] = {}
        for step in range(min_steps):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for city in source_cities:
                x, y, A = source_data[city]
                idx = city_batches[city][step]
                xb = prepare_x(cut_input(x[idx].to(device), args.input_len))
                yb_norm = norm_y(y[idx].to(device), source_stats[city])
                A_dev = A.to(device)
                split = max(args.batch_size, int(x.shape[0] * args.source_train_ratio))
                rng = np.random.RandomState(args.seed + epoch * 100000 + step * 100 + len(city))
                ksup = min(args.task_support_windows, split)
                sup_idx = rng.choice(np.arange(split), size=ksup, replace=split < ksup)
                sx = prepare_x(cut_input(x[sup_idx].to(device), args.input_len))
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    A_task, tg_aux = model.build_task_graph(sx, A_dev)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    pred, aux = model(xb, use_target_adapter=False, A_static=A_dev, A_task=A_task)
                    aux.update(tg_aux)
                    loss_pred = prediction_loss(pred, yb_norm)
                    loss = loss_pred / len(source_cities)
                total_loss = total_loss + loss
                epoch_pred_losses.append(float(loss_pred.detach().cpu().item()))
                collect_scalar_diag(aux, diag_store)
            scaler.scale(total_loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        val_loss, _ = source_city_val_loss(
            model, source_data, source_stats, source_cities, args, device
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "source_cities": source_cities,
                    "source_stats": source_stats,
                    "best_val": best_val,
                    "epoch": epoch,
                },
                best_path,
            )
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.source_epochs:
            diag = {k: float(np.mean(v)) for k, v in diag_store.items() if v}
            log(
                f"[SOURCE] {epoch:03d}/{args.source_epochs} "
                f"pred={np.mean(epoch_pred_losses):.6f} val_MAE={val_loss:.6f} "
                f"best={best_val:.6f} A_ent={diag.get('ttg_A_entropy', 0.0):.4f} "
                f"task_ent={diag.get('task_A_entropy', 0.0):.4f} "
                f"mix={diag.get('task_mix_eta_mean', 0.0):.3f}"
            )
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    return best_path


def finetune_target(model, target_train, target_val, target_stats, args, device, save_dir: Path):
    use_target_adapter = True
    model.set_target_adapter(adapter_ratio=args.adapter_ratio, alpha_init=args.alpha_init)
    model.to(device)
    keys = ["target_adapter", "head", "raw_graph_gate", "graph_residual_branch"]
    total, trainable = set_finetune_params(model, keys)
    log(
        f"[FINETUNE] mode=adapter_head trainable={trainable}/{total} ({100.0 * trainable / total:.4f}%) keys={keys}"
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters in finetune stage.")
    optimizer = torch.optim.Adam(params, lr=args.target_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    best_path = save_dir / "finetuned_best.pt"
    xtr, ytr, Atr = target_train
    xva, yva, _ = target_val
    A_dev = Atr.to(device)
    target_A_task, target_tg_aux = build_task_graph_from_support(model, xtr, Atr, args, device)
    if target_A_task is not None:
        diag0 = {
            k: float(v.detach().cpu().item())
            for k, v in target_tg_aux.items()
            if torch.is_tensor(v) and v.numel() == 1
        }
        log(
            f"[TASK-GRAPH][TARGET] ent={diag0.get('task_A_entropy', 0.0):.4f} max={diag0.get('task_A_max_mean', 0.0):.4f} qk={diag0.get('task_qk_abs_mean', 0.0):.4f} rel={diag0.get('task_rel_bias_abs_mean', 0.0):.4f}"
        )
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        batches = make_batches(
            xtr.shape[0],
            args.batch_size,
            shuffle=True,
            seed=args.seed + 9000 + epoch,
            drop_last=False,
        )
        losses = []
        diag_store: Dict[str, List[float]] = {}
        for idx in batches:
            xb = prepare_x(cut_input(xtr[idx].to(device), args.input_len))
            yb_norm = norm_y(ytr[idx].to(device), target_stats)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred, aux = model(
                    xb, use_target_adapter=use_target_adapter, A_static=A_dev, A_task=target_A_task
                )
                loss = prediction_loss(pred, yb_norm)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
            collect_scalar_diag(aux, diag_store)
        val_loss, _, _, _, val_diag = evaluate_city(
            model,
            xva,
            yva,
            args.batch_size,
            device,
            input_len=args.input_len,
            use_target_adapter=use_target_adapter,
            y_stats=target_stats,
            A=A_dev,
            A_task=target_A_task,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "target_stats": target_stats,
                    "best_val": best_val,
                    "epoch": epoch,
                    "use_target_adapter": use_target_adapter,
                },
                best_path,
            )
        else:
            patience += 1
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.finetune_epochs:
            diag = {k: float(np.mean(v)) for k, v in diag_store.items() if v}
            log(
                f"[FINETUNE] {epoch:03d}/{args.finetune_epochs} "
                f"train={np.mean(losses):.6f} val_MAE={val_loss:.6f} "
                f"best={best_val:.6f}@{best_epoch} "
                f"patience={patience}/{args.early_stop_patience} "
                f"A_ent={val_diag.get('ttg_A_entropy', 0.0):.4f} "
                f"mix={val_diag.get('task_mix_eta_mean', 0.0):.3f}"
            )
        if patience >= args.early_stop_patience:
            log(f"[FINETUNE] early_stop epoch={epoch} best={best_val:.6f}@{best_epoch}")
            break
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    return (best_path, best_epoch, best_val, use_target_adapter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CrossSRM publication training")
    parser.add_argument("--config_filename", default="./configs/config_pems.yaml")
    parser.add_argument("--target_city", default="pems-bay")
    parser.add_argument("--data_list", default="chengdu_shenzhen_metr")
    parser.add_argument("--target_days", type=int, default=3)
    parser.add_argument("--support_train_days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_root", default="./save/crosssrm")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--input_len", type=int, default=288)
    parser.add_argument("--source_epochs", type=int, default=100)
    parser.add_argument("--finetune_epochs", type=int, default=200)
    parser.add_argument("--source_lr", type=float, default=0.001)
    parser.add_argument("--target_lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--source_train_ratio", type=float, default=0.8)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--patch_len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter_ratio", type=int, default=4)
    parser.add_argument("--alpha_init", type=float, default=0.01)
    parser.add_argument(
        "--task_support_windows",
        type=int,
        default=16,
        help="Number of support windows used to generate task graph in source pseudo-tasks.",
    )
    parser.add_argument(
        "--task_graph_topm",
        type=int,
        default=24,
        help="Candidate edges per node before final task top-k.",
    )
    parser.add_argument("--task_graph_dim", type=int, default=64)
    parser.add_argument("--task_graph_rel_hidden", type=int, default=32)
    parser.add_argument("--task_graph_beta", type=float, default=1.0)
    parser.add_argument("--task_graph_qk_scale", type=float, default=1.0)
    parser.add_argument("--task_graph_static_candidate_weight", type=float, default=0.5)
    parser.add_argument(
        "--task_graph_temperature",
        type=float,
        default=0.05,
        help="Temperature used only by the support-conditioned task graph softmax.",
    )
    parser.add_argument(
        "--task_graph_mix",
        type=float,
        default=0.3,
        help="Task graph mixing ratio eta. Final graph=(1-eta)*stable + eta*task when mode=fixed.",
    )
    parser.add_argument("--graph_branch_hidden", type=int, default=128)
    parser.add_argument("--graph_branch_dropout", type=float, default=0.1)
    parser.add_argument("--graph_gate_init", type=float, default=0.01)
    parser.add_argument("--graph_gate_max", type=float, default=0.3)
    parser.add_argument("--graph_topk", type=int, default=8)
    parser.add_argument("--graph_temperature", type=float, default=0.2)
    parser.add_argument("--mv_max_lag", type=int, default=3)
    parser.add_argument(
        "--mv_init_weights",
        type=str,
        default="0.2,0.4,0.3,0.1",
        help="corr,diff,lag,stats initial weights",
    )
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_predictions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.amp = not args.no_amp
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    config = load_yaml(args.config_filename)
    task_args = copy.deepcopy(config["task"]["maml"])
    task_args["batch_size"] = int(args.batch_size)
    task_args["test_dataset"] = args.target_city
    if args.input_len <= 0:
        args.input_len = int(task_args["his_num"])
    source_cities = [c for c in get_data_list(args.data_list) if c != args.target_city]
    if not source_cities:
        raise RuntimeError("source_cities is empty. Check --data_list and --target_city.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "crosssrm"
    save_dir = Path(args.save_root) / f"{tag}_target_{args.target_city}_seed_{args.seed}_{stamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = TeeLogger(save_dir / "run.log")
    log(f"Device={device}")
    log(f"Sources={source_cities} -> Target={args.target_city} | seed={args.seed}")
    log(
        f"Model=CrossSRM | topk={args.graph_topk} graph_temperature={args.graph_temperature} task_graph_temperature={args.task_graph_temperature} task_graph_mix={args.task_graph_mix} graph_gate_max={args.graph_gate_max}"
    )
    log(f"SaveDir={save_dir}")
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    data_args = config["data"]
    source_ds = traffic_dataset(
        data_args,
        task_args,
        data_list=source_cities,
        stage="source_train",
        test_data=args.target_city,
        add_target=False,
        target_days=args.target_days,
    )
    source_data = {c: get_city_tensors(source_ds, c) for c in source_cities}
    source_stats = {
        c: make_dataset_norm_stats(source_ds, c, source_data[c][1]) for c in source_cities
    }
    log(
        "Source data: "
        + ", ".join(
            [
                f"{c}:x{tuple(source_data[c][0].shape)} A{tuple(source_data[c][2].shape)}"
                for c in source_cities
            ]
        )
    )
    target_support_ds = traffic_dataset(
        data_args,
        task_args,
        data_list=source_cities,
        stage="target_maml",
        test_data=args.target_city,
        add_target=False,
        target_days=args.target_days,
        norm_scope="target_support",
    )
    target_test_ds = traffic_dataset(
        data_args,
        task_args,
        data_list=source_cities,
        stage="test",
        test_data=args.target_city,
        add_target=False,
        target_days=args.target_days,
        norm_scope="target_support",
    )
    tx, ty, tA = get_city_tensors(target_support_ds, args.target_city)
    test_x, test_y, test_A = get_city_tensors(target_test_ds, args.target_city)
    train_idx, val_idx = split_target_support_indices(
        num_windows=tx.shape[0],
        his_num=int(task_args["his_num"]),
        pred_num=int(task_args["pred_num"]),
        target_days=args.target_days,
        support_train_days=args.support_train_days,
        stride=12,
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(f"Bad support split: train={len(train_idx)}, val={len(val_idx)}")
    target_train = (tx[train_idx], ty[train_idx], tA)
    target_val = (tx[val_idx], ty[val_idx], tA)
    target_test = (test_x, test_y, test_A)
    target_stats = make_dataset_norm_stats(
        target_support_ds, args.target_city, torch.cat([target_train[1], target_val[1]], dim=0)
    )
    log(
        f"Target support: train={len(train_idx)} val={len(val_idx)} test={test_x.shape[0]} | y_mean={target_stats['mean']:.4f} y_std={target_stats['std']:.4f}"
    )
    model = CrossSRM(
        input_len=args.input_len,
        in_dim=1,
        out_dim=int(task_args["pred_num"]),
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        adapter_ratio=args.adapter_ratio,
        alpha_init=args.alpha_init,
        graph_branch_hidden=args.graph_branch_hidden,
        graph_branch_dropout=args.graph_branch_dropout,
        graph_gate_init=args.graph_gate_init,
        graph_gate_max=args.graph_gate_max,
        task_graph_topm=args.task_graph_topm,
        task_graph_dim=args.task_graph_dim,
        task_graph_rel_hidden=args.task_graph_rel_hidden,
        task_graph_beta=args.task_graph_beta,
        task_graph_qk_scale=args.task_graph_qk_scale,
        task_graph_static_candidate_weight=args.task_graph_static_candidate_weight,
        task_graph_temperature=args.task_graph_temperature,
        task_graph_mix=args.task_graph_mix,
        graph_topk=args.graph_topk,
        graph_temperature=args.graph_temperature,
        mv_max_lag=args.mv_max_lag,
        mv_init_weights=tuple((float(x) for x in args.mv_init_weights.split(","))),
    )
    log(f"Params={sum((p.numel() for p in model.parameters()))} | patches={model.num_patches}")
    pretrain_best = train_source(
        model, source_data, source_stats, source_cities, args, device, save_dir
    )
    log(f"Best source checkpoint: {pretrain_best}")
    finetuned_best, best_epoch, best_val, use_target_adapter = finetune_target(
        model, target_train, target_val, target_stats, args, device, save_dir
    )
    log(f"Best finetune checkpoint: {finetuned_best} | val_MAE={best_val:.6f}@{best_epoch}")
    final_A_task, final_tg_aux = build_task_graph_from_support(
        model, target_train[0], target_train[2], args, device
    )
    final_loss, final_metrics, pred_np, y_np, final_diag = evaluate_city(
        model,
        target_test[0],
        target_test[1],
        args.batch_size,
        device,
        input_len=args.input_len,
        use_target_adapter=use_target_adapter,
        y_stats=target_stats,
        A=target_test[2],
        A_task=final_A_task,
    )
    print_final_metrics(final_metrics)
    avg = average_metrics(final_metrics)
    log(
        f"[SUMMARY] MAE={avg['MAE']:.6f} RMSE={avg['RMSE']:.6f} MSE={avg['MSE']:.6f} MAPE={avg['MAPE']:.6f} | test_loss={final_loss:.6f} | diag={final_diag}"
    )
    if args.save_predictions:
        np.savez_compressed(save_dir / "final_predictions.npz", pred=pred_np, y=y_np)
    with open(save_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": final_metrics,
                "avg": avg,
                "diag": final_diag,
                "target_stats": target_stats,
                "source_stats": source_stats,
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log(f"Saved metrics: {save_dir / 'final_metrics.json'}")
    log("Done.")


if __name__ == "__main__":
    main()

"""Training loop. Keep CLI concerns in train.py."""
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import OCRConfig
from .dataset import BaybayinDataset, collate_ocr_batch, read_label_rows
from .decoding import greedy_decode_batch
from .metrics import character_confusion_matrix, corpus_cer, exact_match_accuracy
from .model import MobileBaybayinOCR
from .vocabulary import BaybayinVocabulary


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _append_jsonl(path: Optional[str | Path], record: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_rows(rows, val_split: float, seed: int):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if len(rows) < 3 or val_split <= 0:
        return indices, []
    n_val = min(len(rows) - 1, max(1, int(round(len(rows) * val_split))))
    return indices[n_val:], indices[:n_val]


def _evaluate(model, loader, vocabulary, device, ctc_loss):
    model.eval()
    total_loss = 0.0
    batches = 0
    references, hypotheses = [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            logits = model(images)
            actual_input_lengths = input_lengths.clamp(max=logits.shape[1])
            log_probs = logits.log_softmax(-1).transpose(0, 1).contiguous()
            loss = ctc_loss(log_probs, targets, actual_input_lengths, target_lengths)
            total_loss += float(loss.item())
            batches += 1
            hypotheses.extend(greedy_decode_batch(logits, actual_input_lengths.tolist(), vocabulary))
            references.extend(batch["texts"])
    if not batches:
        return {"val_loss": float("nan"), "val_cer": float("nan"), "word_accuracy": float("nan")}
    confusion_labels, confusion_matrix = character_confusion_matrix(
        references, hypotheses, vocabulary.characters)
    return {
        "val_loss": total_loss / batches,
        "val_cer": corpus_cer(references, hypotheses),
        "word_accuracy": exact_match_accuracy(references, hypotheses),
        "confusion_labels": confusion_labels,
        "confusion_matrix": confusion_matrix,
    }


def train_model(config: OCRConfig, dataset_root: str | Path = "dataset",
                output_dir: str | Path = "checkpoints", metrics_jsonl: Optional[str | Path] = None,
                resume: Optional[str | Path] = None) -> Dict[str, object]:
    config.validate()
    _seed_everything(config.seed)
    device = torch.device(config.resolved_device())
    data_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_label_rows(data_root)
    vocabulary = BaybayinVocabulary.from_texts(text for _, text in rows)
    train_indices, val_indices = _split_rows(rows, config.val_split, config.seed)
    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]
    train_dataset = BaybayinDataset(data_root, train_rows, vocabulary, config.image_height,
                                    config.image_width, augment=config.augment, seed=config.seed)
    val_dataset = BaybayinDataset(data_root, val_rows, vocabulary, config.image_height,
                                  config.image_width, augment=False, seed=config.seed)
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=use_cuda,
        collate_fn=collate_ocr_batch, drop_last=False,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=use_cuda,
        collate_fn=collate_ocr_batch, persistent_workers=config.num_workers > 0,
    ) if val_rows else None

    model = MobileBaybayinOCR(vocabulary.num_classes, config.feature_channels, config.temporal_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    start_epoch = 1
    best_cer = float("inf")
    best_loss = float("inf")
    global_step = 0

    if resume:
        checkpoint = torch.load(str(resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_cer = float(checkpoint.get("best_cer", best_cer))
        global_step = int(checkpoint.get("global_step", 0))
        # Resume must use the exact label vocabulary stored with the checkpoint.
        checkpoint_vocab = checkpoint.get("vocabulary")
        if checkpoint_vocab and checkpoint_vocab != vocabulary.to_dict():
            raise ValueError("Dataset-derived vocabulary does not match checkpoint vocabulary. Do not resume with changed labels.")

    print(f"Device: {device}; samples: {len(rows)}; train: {len(train_rows)}; val: {len(val_rows)}")
    print(f"Classes incl. CTC blank: {vocabulary.num_classes}; parameters: {model.count_parameters():,}")
    _append_jsonl(metrics_jsonl, {"kind": "start", "samples": len(rows), "train_samples": len(train_rows),
                                  "val_samples": len(val_rows), "parameters": model.count_parameters()})

    epochs_without_improvement = 0
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_batches = 0
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            input_lengths = input_lengths.clamp(max=logits.shape[1])
            log_probs = logits.log_softmax(-1).transpose(0, 1).contiguous()
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite CTC loss at epoch={epoch}, batch={batch_idx}.")
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            global_step += 1
            loss_value = float(loss.item())
            running_loss += loss_value
            running_batches += 1
            epoch_loss_sum += loss_value
            epoch_batches += 1
            if batch_idx % max(1, config.log_every_n_batches) == 0 or batch_idx == len(train_loader):
                _append_jsonl(metrics_jsonl, {
                    "kind": "batch", "epoch": epoch, "batch": batch_idx,
                    "batches_per_epoch": len(train_loader), "global_step": global_step,
                    "train_loss": running_loss / max(1, running_batches),
                    "epoch_progress": batch_idx / max(1, len(train_loader)),
                })
                print(f"Epoch {epoch}/{config.epochs} | batch {batch_idx}/{len(train_loader)} "
                      f"| loss {running_loss / max(1, running_batches):.4f}", flush=True)
                running_loss = 0.0
                running_batches = 0

        # Training epoch loss comes from the actual optimizer steps; avoid a second
        # full pass over the train set just to compute a display metric.
        train_loss = epoch_loss_sum / max(1, epoch_batches)
        if val_loader:
            val_metrics = _evaluate(model, val_loader, vocabulary, device, ctc_loss)
            val_loss = val_metrics["val_loss"]
            val_cer = val_metrics["val_cer"]
            word_accuracy = val_metrics["word_accuracy"]
            confusion_labels = val_metrics["confusion_labels"]
            confusion_matrix = val_metrics["confusion_matrix"]
            monitored = val_cer
            improved = math.isfinite(val_cer) and val_cer < best_cer
        else:
            val_loss, val_cer, word_accuracy = float("nan"), float("nan"), float("nan")
            confusion_labels, confusion_matrix = [], []
            monitored = train_loss
            improved = math.isfinite(train_loss) and train_loss < best_loss

        scheduler.step()
        duration = time.perf_counter() - epoch_start
        record = {
            "kind": "epoch", "epoch": epoch, "global_step": global_step,
            "train_loss": float(train_loss), "val_loss": float(val_loss),
            "val_cer": float(val_cer), "word_accuracy": float(word_accuracy),
            "confusion_labels": confusion_labels, "confusion_matrix": confusion_matrix,
            "lr": float(optimizer.param_groups[0]["lr"]), "seconds": float(duration),
        }
        _append_jsonl(metrics_jsonl, record)
        print(f"Epoch {epoch}/{config.epochs} complete | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_CER={val_cer:.4f} exact_accuracy={word_accuracy:.4f} "
              f"time={duration:.1f}s", flush=True)

        if val_loader:
            if improved:
                best_cer = val_cer
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            if improved:
                best_loss = train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": {"feature_channels": config.feature_channels, "temporal_blocks": config.temporal_blocks,
                             "image_height": config.image_height, "image_width": config.image_width},
            "training_config": config.to_dict(),
            "vocabulary": vocabulary.to_dict(),
            "best_cer": best_cer,
            "best_loss": best_loss,
            "global_step": global_step,
            "metrics": record,
        }
        torch.save(checkpoint_data, output_dir / "last_model.pt")
        if improved:
            torch.save(checkpoint_data, output_dir / "best_model.pt")
        if config.patience > 0 and epochs_without_improvement >= config.patience:
            print(f"Early stopping: no improvement for {config.patience} epochs.", flush=True)
            break

    final_path = output_dir / "last_model.pt"
    return {"checkpoint": str(final_path), "best_checkpoint": str(output_dir / "best_model.pt"),
            "vocabulary": vocabulary.to_dict(), "parameters": model.count_parameters(), "device": str(device)}

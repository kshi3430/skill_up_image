import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def device_name():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def make_model(classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    predictions, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if optimizer:
                loss.backward(); optimizer.step()
        pred = logits.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct += (pred == labels).sum().item(); count += labels.size(0)
        predictions.extend(pred.cpu().tolist()); targets.extend(labels.cpu().tolist())
    return total_loss / count, correct / count, targets, predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data_clean"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    device = device_name(); print("device:", device)

    train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(.75, 1)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(12), transforms.ColorJitter(.15,.15,.15,.05), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train_ds = datasets.ImageFolder(args.data / "train", train_tf)
    val_ds = datasets.ImageFolder(args.data / "validation", eval_tf)
    test_ds = datasets.ImageFolder(args.data / "test", eval_tf)
    kwargs = {"batch_size": args.batch_size, "num_workers": 2, "pin_memory": device == "cuda"}
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes))
    weights = torch.tensor(len(train_ds) / (len(counts) * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
    model = make_model(len(train_ds.classes), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader, loss_fn, device)
        print(f"{epoch:02d} train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, args.output / "best.pt")

    ckpt = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    _, test_acc, y_true, y_pred = run_epoch(model, test_loader, loss_fn, device)
    report = classification_report(y_true, y_pred, target_names=train_ds.classes, output_dict=True, zero_division=0)
    result = {"test_accuracy": test_acc, "classes": train_ds.classes, "classification_report": report, "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"test accuracy={test_acc:.3f}; metrics: {args.output/'metrics.json'}")


if __name__ == "__main__":
    main()

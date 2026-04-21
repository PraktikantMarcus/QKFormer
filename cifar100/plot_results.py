import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = "output/train/20260416-114145-vitsnn-32/summary.csv"

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

epochs     = [int(float(r["epoch"])) for r in rows]
train_loss = [float(r["train_loss"]) for r in rows]
eval_loss  = [float(r["eval_loss"])  for r in rows]
top1       = [float(r["eval_top1"])  for r in rows]
top5       = [float(r["eval_top5"])  for r in rows]

best_epoch = epochs[top1.index(max(top1))]
best_top1  = max(top1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("QKFormer (QKTA + QKCA) — CIFAR-100 (400 epochs)", fontsize=14, fontweight="bold")

# ── Loss ────────────────────────────────────────────────────────────────────
ax1.plot(epochs, train_loss, label="Train loss", color="#4C72B0", linewidth=1.5)
ax1.plot(epochs, eval_loss,  label="Val loss",   color="#DD8452", linewidth=1.5)
ax1.set_ylabel("Loss")
ax1.legend(framealpha=0.7)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.grid(True, which="major", linestyle="--", alpha=0.4)
ax1.grid(True, which="minor", linestyle=":",  alpha=0.2)

# ── Accuracy ─────────────────────────────────────────────────────────────────
ax2.plot(epochs, top1, label="Top-1 acc", color="#55A868", linewidth=1.5)
ax2.plot(epochs, top5, label="Top-5 acc", color="#C44E52", linewidth=1.5, linestyle="--")
ax2.axvline(best_epoch, color="#55A868", linestyle=":", linewidth=1, alpha=0.7)
ax2.scatter([best_epoch], [best_top1], color="#55A868", s=60, zorder=5)
ax2.annotate(
    f"Best: {best_top1:.2f}% @ ep {best_epoch}",
    xy=(best_epoch, best_top1),
    xytext=(best_epoch - 80, best_top1 - 6),
    fontsize=11,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.5),
    color="#55A868",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#55A868", linewidth=1.2),
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend(framealpha=0.7)
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.grid(True, which="major", linestyle="--", alpha=0.4)
ax2.grid(True, which="minor", linestyle=":",  alpha=0.2)

plt.tight_layout()
out = "output/train/20260416-114145-vitsnn-32/training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
print(f"Best top-1: {best_top1:.2f}%  at epoch {best_epoch}")
plt.show()

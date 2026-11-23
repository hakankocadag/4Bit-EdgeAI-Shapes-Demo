"""
=============================================================================
  28x28 4-Bit Shape Classifier -- Training & 4-Bit Quantization Pipeline
  Classes: 0 = Square (00), 1 = Triangle (01), 2 = Circle (10)

  This script:
  1. Generates synthetic 28x28 (784-pixel) Square, Triangle, and Circle datasets
     using 8x8 supersampled anti-aliasing for smooth sub-pixel edges
  2. Trains a single-layer linear classifier: y = W * x + b
  3. Quantizes weights to 4-bit signed ([-8, +7]) and biases to 24-bit signed
  4. Runs a bit-exact hardware Golden Model check (ReLU + Argmax)
  5. Writes weights.hex, biases.hex, acts.hex, and test_1..test_6.hex to
     2_Simulation/ and saves 28x28 PNG visualizations
=============================================================================
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Fixed Hardware Parameters (must match Verilog top_accelerator.v) ---
IMG_SIZE = 28
VECTOR_LENGTH = IMG_SIZE * IMG_SIZE  # 784
NUM_NEURONS = 3                      # 0=Square, 1=Triangle, 2=Circle
ACC_WIDTH = 24                       # Bias and accumulator bit-width
NUM_TRAIN_PER_CLASS = 600
NUM_VAL_PER_CLASS = 150
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-3
EPOCHS = 250
BATCH_SIZE = 64
QUANT_SCALE = 7.0                    # [0.0, 1.0] -> [0, +7] (4-bit signed [-8, +7])
SUBSAMPLE = 8                        # 8x8 supersampling for anti-aliased edges

CLASS_NAMES = ["Square (00)", "Triangle (01)", "Circle (10)"]

np.random.seed(42)

# High-resolution 224x224 sub-pixel grid covering [-0.5, 27.5]
_sub_coords = (np.arange(IMG_SIZE * SUBSAMPLE) + 0.5) / SUBSAMPLE - 0.5
HR_Y, HR_X = np.meshgrid(_sub_coords, _sub_coords, indexing="ij")
CENTER = (IMG_SIZE - 1) / 2.0  # 13.5


# ============================================================================
# SECTION 1: Synthetic 28x28 Anti-Aliased Shape Generation
# ============================================================================
def _downsample_coverage(hr_mask, amplitude):
    """Average 8x8 sub-pixels per pixel for smooth anti-aliased coverage."""
    coverage = hr_mask.astype(float).reshape(
        IMG_SIZE, SUBSAMPLE, IMG_SIZE, SUBSAMPLE
    ).mean(axis=(1, 3))
    return coverage * amplitude


def _apply_noise_and_dropout(img, noise_std, dropout_prob):
    """Apply optional random pixel corruption and additive Gaussian noise."""
    if dropout_prob > 0.0:
        drop_mask = np.random.rand(IMG_SIZE, IMG_SIZE) < dropout_prob
        corrupt_vals = np.random.uniform(-0.3, 0.3, size=(IMG_SIZE, IMG_SIZE))
        img = np.where(drop_mask, corrupt_vals, img)
    if noise_std > 0.0:
        img = img + np.random.normal(0.0, noise_std, size=(IMG_SIZE, IMG_SIZE))
    return img.reshape(-1)


def generate_square(noise_std=0.15, dropout_prob=0.0, shift=None, half_side=None, amp=None):
    """Generate a 28x28 anti-aliased filled Square."""
    dx, dy = shift if shift is not None else np.random.uniform(-2.0, 2.0, size=2)
    r = half_side if half_side is not None else np.random.uniform(7.0, 10.0)
    fg = amp if amp is not None else np.random.uniform(0.75, 1.0)
    cx, cy = CENTER + dx, CENTER + dy
    hr_mask = (np.abs(HR_X - cx) <= r) & (np.abs(HR_Y - cy) <= r)
    img = _downsample_coverage(hr_mask, fg)
    return _apply_noise_and_dropout(img, noise_std, dropout_prob)


def generate_triangle(noise_std=0.15, dropout_prob=0.0, shift=None, half_h=None, half_w=None, amp=None):
    """Generate a 28x28 anti-aliased upward-pointing filled isosceles Triangle."""
    dx, dy = shift if shift is not None else np.random.uniform(-2.0, 2.0, size=2)
    h = half_h if half_h is not None else np.random.uniform(7.5, 10.5)
    w = half_w if half_w is not None else np.random.uniform(7.5, 10.5)
    fg = amp if amp is not None else np.random.uniform(0.75, 1.0)
    cx, cy = CENTER + dx, CENTER + dy
    top_y = cy - h
    bot_y = cy + h
    rel_y = (HR_Y - top_y) / max(bot_y - top_y, 1e-6)
    hr_mask = (HR_Y >= top_y) & (HR_Y <= bot_y) & (np.abs(HR_X - cx) <= w * rel_y)
    img = _downsample_coverage(hr_mask, fg)
    return _apply_noise_and_dropout(img, noise_std, dropout_prob)


def generate_circle(noise_std=0.15, dropout_prob=0.0, shift=None, radius=None, amp=None):
    """Generate a 28x28 anti-aliased filled Circle."""
    dx, dy = shift if shift is not None else np.random.uniform(-2.0, 2.0, size=2)
    r = radius if radius is not None else np.random.uniform(7.0, 10.5)
    fg = amp if amp is not None else np.random.uniform(0.75, 1.0)
    cx, cy = CENTER + dx, CENTER + dy
    dist2 = (HR_X - cx) ** 2 + (HR_Y - cy) ** 2
    hr_mask = dist2 <= (r ** 2)
    img = _downsample_coverage(hr_mask, fg)
    return _apply_noise_and_dropout(img, noise_std, dropout_prob)


def build_dataset(per_class):
    """Build a balanced, shuffled synthetic dataset across all 3 shape classes."""
    xs, ys = [], []
    for _ in range(per_class):
        drop = np.random.uniform(0.0, 0.15)
        xs.append(generate_square(noise_std=0.15, dropout_prob=drop))
        ys.append(0)
        xs.append(generate_triangle(noise_std=0.15, dropout_prob=drop))
        ys.append(1)
        xs.append(generate_circle(noise_std=0.15, dropout_prob=drop))
        ys.append(2)
    xs = np.array(xs)
    ys = np.array(ys)
    perm = np.random.permutation(len(ys))
    return xs[perm], ys[perm]


def quantize_input_4bit(signal):
    """Quantize pixel vector to 4-bit signed integers in [-8, +7]."""
    return np.clip(np.round(signal * QUANT_SCALE), -8, 7).astype(int)


# ============================================================================
# SECTION 2: Dataset Generation & 4-Bit Input Quantization
# ============================================================================
print("Generating 28x28 anti-aliased shape training and validation datasets...")
X_train_raw, y_train = build_dataset(NUM_TRAIN_PER_CLASS)
X_val_raw, y_val = build_dataset(NUM_VAL_PER_CLASS)

X_train_q = quantize_input_4bit(X_train_raw)
X_val_q = quantize_input_4bit(X_val_raw)

print(f"  Training samples  : {len(y_train)} ({NUM_TRAIN_PER_CLASS} per class)")
print(f"  Validation samples: {len(y_val)} ({NUM_VAL_PER_CLASS} per class)")


# ============================================================================
# SECTION 3: Linear Classifier Training (NumPy Softmax + L2 Regularization)
# ============================================================================
print("\nTraining linear classifier...")

W = np.random.randn(NUM_NEURONS, VECTOR_LENGTH) * 0.01
b = np.zeros(NUM_NEURONS)


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(probs, labels):
    n = len(labels)
    return -np.mean(np.log(probs[np.arange(n), labels] + 1e-10))


def one_hot(labels, num_classes):
    oh = np.zeros((len(labels), num_classes))
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


Y_train_oh = one_hot(y_train, NUM_NEURONS)

for epoch in range(EPOCHS):
    perm = np.random.permutation(len(y_train))
    X_shuf = X_train_q[perm].astype(float)
    Y_shuf = Y_train_oh[perm]
    y_shuf = y_train[perm]

    epoch_loss = 0.0
    num_batches = 0

    for start in range(0, len(y_train), BATCH_SIZE):
        X_batch = X_shuf[start:start + BATCH_SIZE]
        Y_batch = Y_shuf[start:start + BATCH_SIZE]
        y_batch = y_shuf[start:start + BATCH_SIZE]

        logits = X_batch @ W.T + b
        probs = softmax(logits)

        loss = cross_entropy_loss(probs, y_batch) + 0.5 * WEIGHT_DECAY * np.sum(W ** 2)
        epoch_loss += loss
        num_batches += 1

        dz = probs - Y_batch
        dW = (dz.T @ X_batch) / len(X_batch) + WEIGHT_DECAY * W
        db = np.mean(dz, axis=0)

        W -= LEARNING_RATE * dW
        b -= LEARNING_RATE * db

    if (epoch + 1) % 50 == 0 or epoch == 0:
        train_preds = np.argmax(X_train_q.astype(float) @ W.T + b, axis=1)
        val_preds = np.argmax(X_val_q.astype(float) @ W.T + b, axis=1)
        train_acc = np.mean(train_preds == y_train) * 100.0
        val_acc = np.mean(val_preds == y_val) * 100.0
        print(
            f"  Epoch {epoch+1:>3}/{EPOCHS} | Loss: {epoch_loss/num_batches:.4f} | "
            f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%"
        )

float_val_acc = np.mean(np.argmax(X_val_q.astype(float) @ W.T + b, axis=1) == y_val) * 100.0
print(f"\n  Float Validation Accuracy: {float_val_acc:.2f}%")


# ============================================================================
# SECTION 4: Quantize Weights to 4-Bit and Biases to 24-Bit
# ============================================================================
# NOTE: W and b are trained directly on quantized 4-bit inputs X_train_q, so
#       W_q * x_q + b_q = round(W_scale * (W * x_q + b)) -> scale bias by W_scale.
print("\nQuantizing weights and biases...")

W_max = max(np.max(np.abs(W)), 1e-10)
W_scale = 7.0 / W_max
W_q = np.clip(np.round(W * W_scale), -8, 7).astype(int)

max_bias = (1 << (ACC_WIDTH - 1)) - 1
min_bias = -(1 << (ACC_WIDTH - 1))
b_q = np.clip(np.round(b * W_scale).astype(int), min_bias, max_bias)

print(f"  Weight scale (W_scale): {W_scale:.6f}")


def hw_forward(x_q_batch):
    """Bit-exact hardware forward pass: MAC + Bias -> ReLU -> Argmax."""
    raw = x_q_batch @ W_q.T + b_q
    relu = np.maximum(0, raw)
    preds = np.argmax(relu, axis=1)
    return raw, relu, preds


_, _, q_train_preds = hw_forward(X_train_q)
_, _, q_val_preds = hw_forward(X_val_q)
q_train_acc = np.mean(q_train_preds == y_train) * 100.0
q_val_acc = np.mean(q_val_preds == y_val) * 100.0

print(f"  Quantized (ReLU+Argmax) Train Accuracy     : {q_train_acc:.2f}%")
print(f"  Quantized (ReLU+Argmax) Validation Accuracy: {q_val_acc:.2f}%")

for n in range(NUM_NEURONS):
    print(f"  Neuron {n} ({CLASS_NAMES[n]}): W_q=[{W_q[n].min()}, {W_q[n].max()}], bias_q={b_q[n]}")


# ============================================================================
# SECTION 5: Write HEX Files to 2_Simulation/
# ============================================================================
sim_dir = Path(__file__).resolve().parent.parent / "2_Simulation"
sim_dir.mkdir(parents=True, exist_ok=True)
print(f"\nWriting HEX files -> {sim_dir}")

weights_path = sim_dir / "weights.hex"
with open(weights_path, "w", encoding="ascii") as f:
    for n in range(NUM_NEURONS):
        for v in range(VECTOR_LENGTH):
            f.write(f"{W_q[n][v] & 0xF:X}\n")
print(f"  weights.hex: {NUM_NEURONS}x{VECTOR_LENGTH} = {NUM_NEURONS * VECTOR_LENGTH} weights (4-bit)")

hex_digits = ACC_WIDTH // 4
bias_mask = (1 << ACC_WIDTH) - 1
biases_path = sim_dir / "biases.hex"
with open(biases_path, "w", encoding="ascii") as f:
    for n in range(NUM_NEURONS):
        hex_val = int(b_q[n]) & bias_mask
        f.write(f"{hex_val:0{hex_digits}X}\n")
        print(f"  Bias {n} ({CLASS_NAMES[n]}): {int(b_q[n]):>8d} -> 0x{hex_val:0{hex_digits}X}")


def make_corrupted_square():
    sq = generate_square(noise_std=0.22, dropout_prob=0.20, shift=(0.5, -0.5), half_side=8.5, amp=0.9)
    sq_2d = sq.reshape(IMG_SIZE, IMG_SIZE)
    sq_2d[4:9, 4:9] = 0.0  # Partial top-left corner occlusion
    return sq_2d.reshape(-1)


acts_signal = make_corrupted_square()
acts_q = quantize_input_4bit(acts_signal)
with open(sim_dir / "acts.hex", "w", encoding="ascii") as f:
    for val in acts_q:
        f.write(f"{val & 0xF:X}\n")
print(f"  acts.hex   : {VECTOR_LENGTH} pixels (4-bit) -- Default test: Corrupted Square")


# ============================================================================
# SECTION 6: Golden Model Verification & 6 Test Scenarios (test_1..test_6.hex)
# ============================================================================
print("\n=== GOLDEN MODEL VERIFICATION (ReLU + Argmax) ===")

test_scenarios = [
    ("Clean Square", 0, generate_square(noise_std=0.0, dropout_prob=0.0, shift=(0.0, 0.0), half_side=8.5, amp=1.0)),
    ("Clean Triangle", 1, generate_triangle(noise_std=0.0, dropout_prob=0.0, shift=(0.0, 0.0), half_h=9.0, half_w=9.0, amp=1.0)),
    ("Clean Circle", 2, generate_circle(noise_std=0.0, dropout_prob=0.0, shift=(0.0, 0.0), radius=9.0, amp=1.0)),
    ("Corrupted / Noisy Square", 0, make_corrupted_square()),
    ("Noisy & Shifted Circle", 2, generate_circle(noise_std=0.25, dropout_prob=0.05, shift=(2.0, -2.0), radius=8.5, amp=0.9)),
    ("Corrupted / Noisy Triangle", 1, generate_triangle(noise_std=0.22, dropout_prob=0.15, shift=(-1.5, 1.0), half_h=8.5, half_w=8.5, amp=0.9)),
]

all_passed = True
test_visual_data = []
for idx, (name, expected_cls, signal) in enumerate(test_scenarios, start=1):
    tq = quantize_input_4bit(signal)
    raw_scores, relu_scores, pred = hw_forward(tq.reshape(1, -1))
    pred_cls = int(pred[0])
    ok = (pred_cls == expected_cls)
    all_passed = all_passed and ok
    status = "PASS" if ok else "FAIL"

    print(
        f"  Test {idx}: {name:<28} | Expected: {CLASS_NAMES[expected_cls]} | "
        f"Predicted: {CLASS_NAMES[pred_cls]} [{status}]"
    )
    print(
        f"          Raw scores  : {raw_scores[0].tolist()} | "
        f"ReLU scores: {relu_scores[0].tolist()}"
    )
    if name in ("Clean Triangle", "Clean Circle"):
        print(f"          Unique 4-bit pixel levels: {np.unique(tq).tolist()}")

    test_visual_data.append((idx, name, tq, pred_cls, relu_scores[0].tolist()))

    with open(sim_dir / f"test_{idx}.hex", "w", encoding="ascii") as f:
        for val in tq:
            f.write(f"{val & 0xF:X}\n")

assert all_passed, "One or more Golden Model tests failed!"
print("\nAll 6/6 Golden Model tests passed and HEX files were generated.")


# ============================================================================
# SECTION 7: 28x28 Visualizations (PNG)
# ============================================================================
script_dir = Path(__file__).resolve().parent

fig_w, axes_w = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
for n, ax in enumerate(axes_w):
    im_w = ax.imshow(W_q[n].reshape(IMG_SIZE, IMG_SIZE), cmap="RdBu_r", vmin=-8, vmax=7)
    ax.set_title(f"Neuron {n}: {CLASS_NAMES[n]}\n(4-bit Weight Template)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig_w.colorbar(im_w, ax=axes_w, shrink=0.85, label="4-bit Weight [-8, +7]")
fig_w.suptitle("Learned 28x28 4-Bit Weight Templates", fontsize=12, fontweight="bold")
weights_png = script_dir / "weight_templates.png"
fig_w.savefig(weights_png, dpi=150)
plt.close(fig_w)

fig_t, axes_t = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)
for ax, (idx, name, tq, pred_cls, relu_sc) in zip(axes_t.flat, test_visual_data):
    im_t = ax.imshow(tq.reshape(IMG_SIZE, IMG_SIZE), cmap="viridis", vmin=-8, vmax=7)
    ax.set_title(
        f"Test {idx}: {name}\nPred: {CLASS_NAMES[pred_cls]} | ReLU: {relu_sc}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
fig_t.colorbar(im_t, ax=axes_t, shrink=0.75, label="4-bit Pixel Value [-8, +7]")
fig_t.suptitle("28x28 Anti-Aliased 4-Bit Test Shapes & Hardware ReLU Scores", fontsize=12, fontweight="bold")
tests_png = script_dir / "test_shapes.png"
fig_t.savefig(tests_png, dpi=150)
plt.close(fig_t)

print(f"Visualizations saved:\n  - {weights_png}\n  - {tests_png}")

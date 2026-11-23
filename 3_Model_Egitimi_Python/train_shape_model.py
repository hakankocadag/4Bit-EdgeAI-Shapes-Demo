"""
=============================================================================
  28x28 4-Bit Sekil Siniflandirici -- Egitim + 4-Bit Kuantizasyon Pipeline'i
  Siniflar: 0 = Kare (00), 1 = Ucgen (01), 2 = Daire (10)

  Bu script:
  1. Sentetik 28x28 (784 piksel) Kare, Ucgen ve Daire egitim/dogrulama verisi uretir
  2. Tek katmanli dogrusal siniflandirici egitir: y = W * x + b
  3. Agirliklari 4-bit'e ([-8, +7]), bias'lari 24-bit'e kuantize eder
  4. Donanimla birebir uyumlu (ReLU + Argmax) Golden Model dogrulamasi yapar
  5. weights.hex, biases.hex, acts.hex ve test_1..test_5.hex dosyalarini
     2_Simulasyon/ klasorune yazar
=============================================================================
"""
from pathlib import Path
import numpy as np

# --- Sabit Parametreler (Verilog top_accelerator.v ile ayni olmali) ---
IMG_SIZE = 28
VECTOR_LENGTH = IMG_SIZE * IMG_SIZE  # 784
NUM_NEURONS = 3                      # 0=Kare, 1=Ucgen, 2=Daire
ACC_WIDTH = 24                       # Bias ve akkumulator genisligi
NUM_TRAIN_PER_CLASS = 600
NUM_VAL_PER_CLASS = 150
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-3
EPOCHS = 250
BATCH_SIZE = 64
QUANT_SCALE = 7.0                    # [-1.0, +1.0] -> [-7, +7] (4-bit [-8, +7])

CLASS_NAMES = ["Kare (00)", "Ucgen (01)", "Daire (10)"]

np.random.seed(42)

# 28x28 piksel koordinat izgara matrisleri (x: sutun [0..27], y: satir [0..27])
GRID_Y, GRID_X = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
CENTER = (IMG_SIZE - 1) / 2.0  # 13.5


# ============================================================================
# BOLUM 1: Sentetik 28x28 Sekil Veri Uretimi
# ============================================================================
def _apply_noise_and_dropout(img, noise_std, dropout_prob):
    """Goruntuye Gauss gurultusu ve rastgele piksel bozulmasi/eksilmesi uygula."""
    if dropout_prob > 0.0:
        drop_mask = np.random.rand(IMG_SIZE, IMG_SIZE) < dropout_prob
        corrupt_vals = np.random.uniform(-0.3, 0.3, size=(IMG_SIZE, IMG_SIZE))
        img = np.where(drop_mask, corrupt_vals, img)
    if noise_std > 0.0:
        img = img + np.random.normal(0.0, noise_std, size=(IMG_SIZE, IMG_SIZE))
    return img.reshape(-1)


def generate_square(noise_std=0.15, dropout_prob=0.0, shift=None, half_side=None, amp=None):
    """28x28 dolu Kare sablonu."""
    dx, dy = shift if shift is not None else np.random.uniform(-2.0, 2.0, size=2)
    r = half_side if half_side is not None else np.random.uniform(7.0, 10.0)
    fg = amp if amp is not None else np.random.uniform(0.75, 1.0)
    cx, cy = CENTER + dx, CENTER + dy
    mask = (np.abs(GRID_X - cx) <= r) & (np.abs(GRID_Y - cy) <= r)
    img = np.where(mask, fg, 0.0)
    return _apply_noise_and_dropout(img, noise_std, dropout_prob)


def generate_triangle(noise_std=0.15, dropout_prob=0.0, shift=None, half_h=None, half_w=None, amp=None):
    """28x28 yukari bakan dolu ikizkenar Ucgen sablonu."""
    dx, dy = shift if shift is not None else np.random.uniform(-2.0, 2.0, size=2)
    h = half_h if half_h is not None else np.random.uniform(7.5, 10.5)
    w = half_w if half_w is not None else np.random.uniform(7.5, 10.5)
    fg = amp if amp is not None else np.random.uniform(0.75, 1.0)
    cx, cy = CENTER + dx, CENTER + dy
    top_y = cy - h
    bot_y = cy + h
    # Yukseklik boyunca tabana dogru dogrusal genisleyen yari genislik
    rel_y = (GRID_Y - top_y) / max(bot_y - top_y, 1e-6)
    mask = (GRID_Y >= top_y) & (GRID_Y <= bot_y) & (np.abs(GRID_X - cx) <= w * rel_y)
    img = np.where(mask, fg, 0.0)
    return _apply_noise_and_dropout(img, noise_std, dropout_prob)


def generate_circle(noise_std=0.15, dropout_prob=0.0, shift=None, radius=None, amp=None):
    """28x28 dolu Daire sablonu."""
    dx, dy = shift if shift is not None else np.random.uniform(-2.0, 2.0, size=2)
    r = radius if radius is not None else np.random.uniform(7.0, 10.5)
    fg = amp if amp is not None else np.random.uniform(0.75, 1.0)
    cx, cy = CENTER + dx, CENTER + dy
    dist2 = (GRID_X - cx) ** 2 + (GRID_Y - cy) ** 2
    mask = dist2 <= (r ** 2)
    img = np.where(mask, fg, 0.0)
    return _apply_noise_and_dropout(img, noise_std, dropout_prob)


def build_dataset(per_class):
    """Her siniftan esit sayida karisik sentetik sekil veri seti olustur."""
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
    """Goruntu vektorunu [-8, +7] 4-bit isaretli tamsayiya kuantize et."""
    return np.clip(np.round(signal * QUANT_SCALE), -8, 7).astype(int)


# ============================================================================
# BOLUM 2: Egitim & Dogrulama Verisi ve 4-Bit Kuantizasyon
# ============================================================================
print("28x28 sekil egitim ve dogrulama verisi uretiliyor...")
X_train_raw, y_train = build_dataset(NUM_TRAIN_PER_CLASS)
X_val_raw, y_val = build_dataset(NUM_VAL_PER_CLASS)

X_train_q = quantize_input_4bit(X_train_raw)
X_val_q = quantize_input_4bit(X_val_raw)

print(f"  Egitim ornek sayisi   : {len(y_train)} (her siniftan {NUM_TRAIN_PER_CLASS})")
print(f"  Dogrulama ornek sayisi: {len(y_val)} (her siniftan {NUM_VAL_PER_CLASS})")


# ============================================================================
# BOLUM 3: Dogrusal Siniflandirici Egitimi (NumPy ile Softmax + L2)
# ============================================================================
print("\nModel egitimi basliyor...")

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
print(f"\n  Float Dogrulama Dogrulugu: {float_val_acc:.2f}%")


# ============================================================================
# BOLUM 4: Agirliklari 4-Bit'e, Bias'lari 24-Bit'e Kuantize Et
# ============================================================================
# NOT: W ve b zaten 4-bit kuantize X_train_q uzerinde egitildigi icin
#      W_q * x_q + b_q = round(W_scale * (W * x_q + b)) -> bias yalnizca W_scale ile carpilir!
print("\nAgirliklar ve biaslar kuantize ediliyor...")

W_max = max(np.max(np.abs(W)), 1e-10)
W_scale = 7.0 / W_max
W_q = np.clip(np.round(W * W_scale), -8, 7).astype(int)

max_bias = (1 << (ACC_WIDTH - 1)) - 1
min_bias = -(1 << (ACC_WIDTH - 1))
b_q = np.clip(np.round(b * W_scale).astype(int), min_bias, max_bias)

print(f"  Agirlik olcegi (W_scale): {W_scale:.6f}")


def hw_forward(x_q_batch):
    """Verilog donanimi ile birebir ayni ileri yayilim: MAC + Bias -> ReLU -> Argmax."""
    raw = x_q_batch @ W_q.T + b_q
    relu = np.maximum(0, raw)
    preds = np.argmax(relu, axis=1)
    return raw, relu, preds


_, _, q_train_preds = hw_forward(X_train_q)
_, _, q_val_preds = hw_forward(X_val_q)
q_train_acc = np.mean(q_train_preds == y_train) * 100.0
q_val_acc = np.mean(q_val_preds == y_val) * 100.0

print(f"  Kuantize (ReLU+Argmax) Egitim Dogrulugu   : {q_train_acc:.2f}%")
print(f"  Kuantize (ReLU+Argmax) Dogrulama Dogrulugu: {q_val_acc:.2f}%")

for n in range(NUM_NEURONS):
    print(f"  Neuron {n} ({CLASS_NAMES[n]}): W_q=[{W_q[n].min()}, {W_q[n].max()}], bias_q={b_q[n]}")


# ============================================================================
# BOLUM 5: HEX Dosyalarini 2_Simulasyon/ Dizinine Yaz
# ============================================================================
sim_dir = Path(__file__).resolve().parent.parent / "2_Simulasyon"
sim_dir.mkdir(parents=True, exist_ok=True)
print(f"\nHEX dosyalari olusturuluyor -> {sim_dir}")

weights_path = sim_dir / "weights.hex"
with open(weights_path, "w", encoding="ascii") as f:
    for n in range(NUM_NEURONS):
        for v in range(VECTOR_LENGTH):
            f.write(f"{W_q[n][v] & 0xF:X}\n")
print(f"  weights.hex: {NUM_NEURONS}x{VECTOR_LENGTH} = {NUM_NEURONS * VECTOR_LENGTH} agirlik (4-bit)")

hex_digits = ACC_WIDTH // 4
bias_mask = (1 << ACC_WIDTH) - 1
biases_path = sim_dir / "biases.hex"
with open(biases_path, "w", encoding="ascii") as f:
    for n in range(NUM_NEURONS):
        hex_val = int(b_q[n]) & bias_mask
        f.write(f"{hex_val:0{hex_digits}X}\n")
        print(f"  Bias {n} ({CLASS_NAMES[n]}): {int(b_q[n]):>8d} -> 0x{hex_val:0{hex_digits}X}")

# Bozuk / Gurultulu Kare olusturucu (hem acts.hex hem test_4.hex icin)
def make_corrupted_square():
    sq = generate_square(noise_std=0.22, dropout_prob=0.20, shift=(0.5, -0.5), half_side=8.5, amp=0.9)
    # Sol ust kosede kismi bozulma (Fikir.pdf: "bozuk bir kare")
    sq_2d = sq.reshape(IMG_SIZE, IMG_SIZE)
    sq_2d[4:9, 4:9] = 0.0
    return sq_2d.reshape(-1)


acts_signal = make_corrupted_square()
acts_q = quantize_input_4bit(acts_signal)
with open(sim_dir / "acts.hex", "w", encoding="ascii") as f:
    for val in acts_q:
        f.write(f"{val & 0xF:X}\n")
print(f"  acts.hex   : {VECTOR_LENGTH} piksel (4-bit) -- Varsayilan test: Bozuk Kare")


# ============================================================================
# BOLUM 6: Golden Model Dogrulamasi ve 5 Test Senaryosu (test_1..test_5.hex)
# ============================================================================
print("\n=== GOLDEN MODEL DOGRULAMA (ReLU + Argmax) ===")

test_scenarios = [
    ("Kusursuz Kare", 0, generate_square(noise_std=0.0, dropout_prob=0.0, shift=(0.0, 0.0), half_side=8.5, amp=1.0)),
    ("Kusursuz Ucgen", 1, generate_triangle(noise_std=0.0, dropout_prob=0.0, shift=(0.0, 0.0), half_h=9.0, half_w=9.0, amp=1.0)),
    ("Kusursuz Daire", 2, generate_circle(noise_std=0.0, dropout_prob=0.0, shift=(0.0, 0.0), radius=9.0, amp=1.0)),
    ("Bozuk / Gurultulu Kare", 0, make_corrupted_square()),
    ("Gurultulu ve Kaymis Daire", 2, generate_circle(noise_std=0.25, dropout_prob=0.05, shift=(2.0, -2.0), radius=8.5, amp=0.9)),
]

all_passed = True
for idx, (name, expected_cls, signal) in enumerate(test_scenarios, start=1):
    tq = quantize_input_4bit(signal)
    raw_scores, relu_scores, pred = hw_forward(tq.reshape(1, -1))
    pred_cls = int(pred[0])
    ok = (pred_cls == expected_cls)
    all_passed = all_passed and ok
    status = "BASARILI" if ok else "HATA"

    print(
        f"  Test {idx}: {name:<28} | Beklenen: {CLASS_NAMES[expected_cls]} | "
        f"Tahmin: {CLASS_NAMES[pred_cls]} [{status}]"
    )
    print(
        f"          Ham skorlar : {raw_scores[0].tolist()} | "
        f"ReLU skorlar: {relu_scores[0].tolist()}"
    )

    with open(sim_dir / f"test_{idx}.hex", "w", encoding="ascii") as f:
        for val in tq:
            f.write(f"{val & 0xF:X}\n")

assert all_passed, "Golden model testlerinden en az biri basarisiz oldu!"
print("\nTum 5/5 Golden Model testi basariyla gecti ve HEX dosyalari yazildi.")

"""
=============================================================================
  EKG Sınıflandırıcı — Eğitim + 4-Bit Kuantizasyon Pipeline'ı
  
  Bu script:
  1. Sentetik EKG eğitim verisi üretir (her sınıftan 500 varyasyon)
  2. Tek katmanlı doğrusal sınıflandırıcı eğitir: y = W·x + b
  3. Ağırlıkları 4-bit'e, bias'ları ACC_WIDTH-bit'e kuantize eder
  4. weights.hex, biases.hex ve acts.hex dosyalarını oluşturur
=============================================================================
"""
import numpy as np

# --- Sabit Parametreler (Verilog ile aynı olmalı) ---
VECTOR_LENGTH = 784
NUM_NEURONS = 3      # 0=Normal, 1=Aritmi(PVC), 2=Gürültü
ACC_WIDTH = 24        # Bias genişliği
NUM_TRAIN = 500       # Her sınıftan eğitim sayısı
LEARNING_RATE = 0.001
EPOCHS = 200

np.random.seed(42)

# ============================================================================
# BÖLÜM 1: Sentetik EKG Veri Üretimi
# ============================================================================
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

x = np.arange(VECTOR_LENGTH)

# --- Temel Şablonlar ---
def generate_normal(x, noise_std=0.1):
    """Normal sinüs ritmi: P, Q, R, S, T dalgaları"""
    # Rastgele varyasyonlar ekle (genlik, pozisyon, genişlik)
    amp_var = np.random.uniform(0.8, 1.2)
    pos_var = np.random.randint(-20, 20)
    signal = (
        gaussian(x, 0.2*amp_var, 300+pos_var, 15) +
        gaussian(x, -0.3*amp_var, 370+pos_var, 5) +
        gaussian(x, 1.5*amp_var, 392+pos_var, 10) +
        gaussian(x, -0.4*amp_var, 415+pos_var, 5) +
        gaussian(x, 0.4*amp_var, 520+pos_var, 30)
    )
    return signal + np.random.normal(0, noise_std, VECTOR_LENGTH)

def generate_arrhythmia(x, noise_std=0.15):
    """PVC Aritmisi: P yok, geniş QRS, ters T"""
    amp_var = np.random.uniform(0.7, 1.3)
    pos_var = np.random.randint(-30, 30)
    signal = (
        gaussian(x, 1.2*amp_var, 370+pos_var, 30) +
        gaussian(x, -1.2*amp_var, 440+pos_var, 40) +
        gaussian(x, -0.3*amp_var, 550+pos_var, 35)
    )
    return signal + np.random.normal(0, noise_std, VECTOR_LENGTH)

def generate_noise(x, noise_std=0.5):
    """Gürültü / bozuk sinyal"""
    return np.random.normal(0, noise_std, VECTOR_LENGTH)

# --- Eğitim Verisi Üret ---
print("Egitim verisi uretiliyor...")
X_train = []
y_train = []

for _ in range(NUM_TRAIN):
    X_train.append(generate_normal(x))
    y_train.append(0)
    
    X_train.append(generate_arrhythmia(x))
    y_train.append(1)
    
    X_train.append(generate_noise(x))
    y_train.append(2)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Karıştır
indices = np.random.permutation(len(y_train))
X_train = X_train[indices]
y_train = y_train[indices]

print(f"  Toplam egitim verisi: {len(y_train)} (her siniftan {NUM_TRAIN})")

# ============================================================================
# BÖLÜM 2: 4-Bit Kuantizasyon Fonksiyonları
# ============================================================================
def quantize_4bit(signal):
    """Sinyali [-8, +7] aralığına kuantize et"""
    max_val = max(np.max(np.abs(signal)), 1e-10)
    scale = 7.0 / max_val
    return np.clip(np.round(signal * scale), -8, 7).astype(int), scale

def quantize_input_4bit(signal, scale):
    """Verilen ölçekle sinyali [-8, +7] aralığına kuantize et"""
    return np.clip(np.round(signal * scale), -8, 7).astype(int)

# Tüm eğitim verisini kuantize et (ortak ölçek)
global_max = np.max(np.abs(X_train))
input_scale = 7.0 / global_max
X_train_q = np.clip(np.round(X_train * input_scale), -8, 7).astype(int)

print(f"  Giris olcegi (input_scale): {input_scale:.6f}")

# ============================================================================
# BÖLÜM 3: Doğrusal Sınıflandırıcı Eğitimi (NumPy ile)
# ============================================================================
# Model: y = W·x + b (W: 3×784, b: 3)
# Softmax + Cross-Entropy Loss

print("\nModel egitimi basliyor...")

# Ağırlıkları küçük rastgele değerlerle başlat
W = np.random.randn(NUM_NEURONS, VECTOR_LENGTH) * 0.01
b = np.zeros(NUM_NEURONS)

def softmax(logits):
    """Numerik olarak kararlı softmax"""
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    """Cross-entropy loss"""
    n = len(labels)
    return -np.mean(np.log(probs[np.arange(n), labels] + 1e-10))

# One-hot encoding
def one_hot(labels, num_classes):
    oh = np.zeros((len(labels), num_classes))
    oh[np.arange(len(labels)), labels] = 1
    return oh

Y_one_hot = one_hot(y_train, NUM_NEURONS)

# Mini-batch SGD eğitimi
batch_size = 64

for epoch in range(EPOCHS):
    # Karıştır
    perm = np.random.permutation(len(y_train))
    X_shuffled = X_train_q[perm].astype(float)
    Y_shuffled = Y_one_hot[perm]
    y_shuffled = y_train[perm]
    
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, len(y_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]
        
        # Forward pass: z = X·W^T + b
        logits = X_batch @ W.T + b
        probs = softmax(logits)
        
        # Loss
        batch_labels = y_shuffled[i:i+batch_size]
        loss = cross_entropy_loss(probs, batch_labels)
        epoch_loss += loss
        num_batches += 1
        
        # Backward pass
        dz = probs - Y_batch  # (batch, 3)
        dW = (dz.T @ X_batch) / len(X_batch)
        db = np.mean(dz, axis=0)
        
        # Gradient descent
        W -= LEARNING_RATE * dW
        b -= LEARNING_RATE * db
    
    if (epoch + 1) % 50 == 0 or epoch == 0:
        # Doğruluk hesapla
        logits_all = X_train_q.astype(float) @ W.T + b
        preds = np.argmax(logits_all, axis=1)
        accuracy = np.mean(preds == y_train) * 100
        print(f"  Epoch {epoch+1:>3}/{EPOCHS} | Loss: {epoch_loss/num_batches:.4f} | Dogruluk: {accuracy:.1f}%")

# Son doğruluk
logits_all = X_train_q.astype(float) @ W.T + b
preds = np.argmax(logits_all, axis=1)
final_accuracy = np.mean(preds == y_train) * 100
print(f"\n  FINAL DOGRULUK: {final_accuracy:.1f}%")

# ============================================================================
# BÖLÜM 4: Ağırlıkları 4-Bit'e Kuantize Et + Bias Hesapla
# ============================================================================
print("\nAgirliklar kuantize ediliyor...")

# Ağırlıkları 4-bit'e kuantize et
W_max = np.max(np.abs(W))
W_scale = 7.0 / W_max
W_q = np.clip(np.round(W * W_scale), -8, 7).astype(int)

# Bias'ı hesapla: kuantize ağırlıklarla gerçek bias arasındaki farkı telafi et
# Kuantize model: y = W_q · x_q + b_q
# b_q, tam sayı cinsinden olmalı (ACC_WIDTH genişliğinde)
# Bias'ı doğrudan ölçeklendirip tam sayıya çevir
bias_scale = W_scale * input_scale  # Ağırlık ve giriş ölçeklerinin çarpımı
b_q = np.round(b * bias_scale).astype(int)

# ACC_WIDTH sınırlarında kırp
max_bias = (1 << (ACC_WIDTH - 1)) - 1    # +8388607
min_bias = -(1 << (ACC_WIDTH - 1))       # -8388608
b_q = np.clip(b_q, min_bias, max_bias)

print(f"  Agirlik olcegi (W_scale): {W_scale:.6f}")
print(f"  Bias olcegi (bias_scale): {bias_scale:.6f}")

# Kuantize modelin doğruluğunu kontrol et
logits_q = X_train_q.astype(float) @ W_q.T + b_q
preds_q = np.argmax(logits_q, axis=1)
q_accuracy = np.mean(preds_q == y_train) * 100
print(f"  Kuantize model dogrulugu: {q_accuracy:.1f}%")

# Her nöronun ağırlık istatistikleri
for n in range(NUM_NEURONS):
    sinif = ["Normal", "Aritmi", "Gurultu"][n]
    print(f"  Neuron {n} ({sinif}): W_q araligi=[{W_q[n].min()}, {W_q[n].max()}], bias_q={b_q[n]}")

# ============================================================================
# BÖLÜM 5: HEX Dosyalarına Yaz (Verilog için)
# ============================================================================
print("\nHEX dosyalari olusturuluyor...")

# --- weights.hex: 3 nöron × 784 ağırlık = 2352 adet 4-bit değer ---
with open("weights.hex", "w") as f:
    for n in range(NUM_NEURONS):
        for v in range(VECTOR_LENGTH):
            f.write(f"{W_q[n][v] & 0xF:X}\n")
print(f"  weights.hex: {NUM_NEURONS}x{VECTOR_LENGTH} = {NUM_NEURONS*VECTOR_LENGTH} agirlik (4-bit)")

# --- biases.hex: 3 adet ACC_WIDTH-bit bias ---
# İkiye tümleyen (two's complement) formatında, ACC_WIDTH/4 hex digit
hex_digits = ACC_WIDTH // 4  # 24-bit = 6 hex digit
mask = (1 << ACC_WIDTH) - 1  # 0xFFFFFF for 24-bit

with open("biases.hex", "w") as f:
    for n in range(NUM_NEURONS):
        hex_val = b_q[n] & mask  # İkiye tümleyen
        f.write(f"{hex_val:0{hex_digits}X}\n")
        sinif = ["Normal", "Aritmi", "Gurultu"][n]
        print(f"  Bias {n} ({sinif}): {b_q[n]:>8d} -> 0x{hex_val:0{hex_digits}X}")

# --- acts.hex: Test hastası (gürültülü aritmi) ---
test_signal = generate_arrhythmia(x, noise_std=0.2)
test_q = quantize_input_4bit(test_signal, input_scale)

with open("acts.hex", "w") as f:
    for val in test_q:
        f.write(f"{val & 0xF:X}\n")
print(f"  acts.hex: {VECTOR_LENGTH} aktivasyon (4-bit) — Test: Gurultulu Aritmi")

# ============================================================================
# BÖLÜM 6: Model Doğrulama — Golden Model Sonuçları
# ============================================================================
print("\n=== GOLDEN MODEL DOGRULAMA ===")
# Test sinyalini kuantize modelden geçir
test_scores_raw = []
test_scores_relu = []
for n in range(NUM_NEURONS):
    dot = int(np.sum(W_q[n] * test_q)) + int(b_q[n])
    test_scores_raw.append(dot)
    test_scores_relu.append(max(0, dot))

predicted = np.argmax(test_scores_relu)
siniflar = ["Normal (00)", "Aritmi (01)", "Gurultu (10)"]

print(f"  Test sinyali: Gurultulu Aritmi")
for n in range(NUM_NEURONS):
    print(f"  Neuron {n} ({siniflar[n]}): ham={test_scores_raw[n]:>8d}, relu={test_scores_relu[n]:>8d}")
print(f"  --> Tahmin: {siniflar[predicted]}")

# Düşman test senaryoları da oluştur
print("\n=== DUSMAN TEST SENARYOLARI ===")
test_scenarios = [
    ("Kusursuz Normal EKG", generate_normal(x, noise_std=0.0)),
    ("Asiri Gurultulu Aritmi (%50)", generate_arrhythmia(x, noise_std=0.5)),
    ("Kablo Kopuk (Sadece Gurultu)", generate_noise(x, noise_std=1.0)),
    ("Zayif Kalp Atisi", generate_normal(x, noise_std=0.02) * 0.1),
    ("Zaman Kaymasi (100 birim)", 
        gaussian(x, 0.2, 400, 15) + gaussian(x, -0.3, 470, 5) + 
        gaussian(x, 1.5, 492, 10) + gaussian(x, -0.4, 515, 5) + 
        gaussian(x, 0.4, 620, 30) + np.random.normal(0, 0.1, VECTOR_LENGTH)),
]

for i, (name, signal) in enumerate(test_scenarios):
    tq = quantize_input_4bit(signal, input_scale)
    scores = []
    for n in range(NUM_NEURONS):
        dot = int(np.sum(W_q[n] * tq)) + int(b_q[n])
        scores.append(max(0, dot))
    pred = np.argmax(scores)
    print(f"  Test {i+1}: {name:<35} -> {siniflar[pred]}")
    
    # Düşman test dosyası oluştur
    with open(f"test_{i+1}.hex", "w") as f:
        for val in tq:
            f.write(f"{val & 0xF:X}\n")

print(f"\nTum dosyalar basariyla olusturuldu!")
print(f"  - weights.hex  ({NUM_NEURONS*VECTOR_LENGTH} agirlik)")
print(f"  - biases.hex   ({NUM_NEURONS} bias)")
print(f"  - acts.hex     (test verisi)")
print(f"  - test_1.hex ~ test_5.hex (dusman testleri)")

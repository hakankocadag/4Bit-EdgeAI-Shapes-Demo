# 4-Bit Quantized Shape Classification Hardware Accelerator

## Proje Hakkında
Bu proje, harici bir işlemciye (CPU) veya işletim sistemine ihtiyaç duymadan doğrudan donanım seviyesinde çalışan **4-bit Kuantize Yapay Zeka Hızlandırıcısı** tasarımını içerir. Tasarım, **28×28 piksel (784 elemanlı, piksel başına 4-bit işaretli)** giriş görüntülerini analiz ederek üç geometrik şekilden hangisine ait olduğunu sınıflandırır:

- `00` — **Kare**
- `01` — **Üçgen**
- `10` — **Daire**

Mimari; tamamen 4-bit veri yolları, taşma korumalı (Saturation) 24-bit akümülatör, kombinasyonel bias toplama, donanımsal ReLU ve sıralı Argmax karşılaştırıcı birimleri kullanılarak sentezlenebilir Verilog-2001 ile yazılmıştır. 50 MHz saat frekansında (20 ns periyot) tek bir çıkarım (inference) yaklaşık **94.2 µs** içinde tamamlanır.

## Klasör Yapısı

### 1. `1_Chip_RTL`
Çip (ASIC / FPGA) için sentezlenebilir saf Verilog donanım modüllerini içerir:
- `top_accelerator.v`: Ana entegrasyon ve kombinasyonel bias toplama birimi (`z = W·x + b`).
- `control_fsm.v`: Moore tipi durum makinesi (`S_IDLE -> S_FETCH -> S_MAC -> S_RELU -> S_OUTPUT -> S_DONE`).
- `mac_unit.v`: 4-bit işaretli Çarpma-Toplama ve 24-bit Saturation birimi.
- `relu_unit.v`: MUX tabanlı donanımsal ReLU aktivasyon birimi.
- `argmax_unit.v`: Sıralı (Sequential) donanımsal maksimum skor karşılaştırıcısı (`predicted_class`).
- `memory_block.v`: Ağırlık (`weights.hex`) ve 28×28 piksel giriş verisini (`acts.hex`) tutan bellek birimi.
- `agu.v`: Adres Üretici (Address Generation Unit).

### 2. `2_Simulasyon`
Simülasyon testbench'lerini ve kuantize `.hex` bellek dosyalarını içerir:
- `tb_brute_force.v`: 6 farklı 28×28 şekil test senaryosunu (`test_1.hex`–`test_6.hex`) donanımda çalıştıran ve `predicted_class` çıkışını `done` sinyali yükseldiği anda doğrulayan sistem testbench'i.
- `tb_full_brute_force.v`: `mac_unit` (256 tekil çarpım + saturation) ve `relu_unit` birimlerini golden model ile doğrulayan birim testbench'i.
- `weights.hex`: 3 nöron × 784 piksel = 2352 adet 4-bit ağırlık.
- `biases.hex`: 3 adet 24-bit işaretli bias (two's complement hex).
- `acts.hex` ve `test_1.hex`–`test_6.hex`: 784 piksellik 4-bit şekil test vektörleri (Kusursuz Kare, Kusursuz Üçgen, Kusursuz Daire, Bozuk/Gürültülü Kare, Gürültülü ve Kaymış Daire, Bozuk/Gürültülü Üçgen).

### 3. `3_Model_Egitimi_Python`
- `train_shape_model.py`: Sentetik 28×28 Kare, Üçgen ve Daire veri seti üretir, Softmax + Cross-Entropy ile doğrusal sınıflandırıcı eğitir, ağırlıkları 4-bit'e (`[-8, +7]`) ve bias'ları 24-bit'e kuantize eder, donanım golden modeliyle doğrulayıp tüm `.hex` dosyalarını `2_Simulasyon/` klasörüne yazar; ayrıca sunum/rapor için `weight_templates.png` ve `test_shapes.png` görsellerini üretir.

## Nasıl Çalıştırılır?

### 1. Python ile Model Eğitimi ve `.hex` Üretimi
```bash
python 3_Model_Egitimi_Python/train_shape_model.py
```

### 2. Verilog Simülasyonu (Questa / ModelSim veya Icarus Verilog)
`2_Simulasyon` dizini içinde çalıştırın (`.hex` dosyalarının çalışma dizininde bulunması gerekir):

**Questa / ModelSim:**
```bash
cd 2_Simulasyon
vlib work
vlog -work work ../1_Chip_RTL/*.v tb_brute_force.v tb_full_brute_force.v
vsim -c -lib work -do "run -all; quit -f" tb_full_brute_force
vsim -c -lib work -do "run -all; quit -f" tb_brute_force
```

**Icarus Verilog:**
```bash
cd 2_Simulasyon
iverilog -o sim.vvp ../1_Chip_RTL/*.v tb_brute_force.v
vvp sim.vvp
```

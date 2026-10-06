# 4-Bit Edge-AI ECG Classification Hardware Accelerator

## Proje Hakkında (Demo ve Kanıt Sürümü)
Bu proje, Computer Architecture dersi kapsamında sunulan **Özelleşmiş CPU (Hardware Accelerator)** ve **FSM (Durum Makinesi)** donanım mimarisi proje fikrinin **"Demo ve Çalışabilirliğinin Kanıtı (Proof of Concept)"** sürümüdür. Projenin amacı, yapay zeka (Makine Öğrenmesi) algoritmalarını harici bir işlemciye (CPU) veya işletim sistemine ihtiyaç duymadan doğrudan donanım seviyesinde koşturmanın mümkün olduğunu göstermektir. 

Sistem, bir hastadan alınan 784 örnekli EKG sinyalini donanımsal olarak analiz edip hastanın durumunu 3 sınıfa (Normal, Aritmi, Gürültü) ayırmaktadır. Tasarım, tamamen 4-bit veri yolları, Saturation (Taşma Koruması) mantığı ve donanımsal ReLU / Argmax mimarileri kullanılarak Verilog-2001 ile yazılmıştır. Bu demo, önerilen donanım mimarisinin Icarus Verilog ortamında mikrosaniyeler içinde kusursuzca çalıştığını ve hedeflenen donanımsal yapay zeka vizyonunun gerçekçi ve uygulanabilir olduğunu kanıtlamaktadır.

## Klasör Yapısı
Bu proje paketi 3 ana klasörden oluşmaktadır:

### 1. `1_Chip_RTL`
Sadece çipe (ASIC / FPGA) basılacak olan saf donanım kodlarını (Sentezlenebilir Verilog) içerir.
- `top_accelerator.v`: Ana entegrasyon ve kombinasyonel Bias birimi.
- `control_fsm.v`: Sistemin Moore/Mealy Durum Makinesi (Control Unit).
- `mac_unit.v`: 4-bit işaretli Çarpma-Toplama ve Saturation birimi (Datapath).
- `relu_unit.v`: MUX tabanlı donanımsal aktivasyon birimi.
- `argmax_unit.v`: Sıralı (Sequential) donanımsal karşılaştırıcı (Karar Mekanizması).
- `memory_block.v`: Ağırlık ve sensör verilerini tutan ROM/RAM bellek birimi.
- `agu.v`: Adres Üretici (Address Generation Unit).

### 2. `2_Simulasyon`
Çipi ModelSim veya Icarus Verilog gibi ortamlarda test etmek ve dalga formlarını (Waveforms) görmek için gereken her şeyi içerir. 
- Çip kodlarına ek olarak `tb_full_brute_force.v` (Testbench) dosyası bulunur.
- Sistemin hafızasına yüklenecek olan eğitilmiş katsayılar (`weights.hex`, `biases.hex`) ve test hastalarına ait veriler (`test_*.hex`) buradadır. *Not: Simülasyon yaparken bu hex dosyalarının RTL kodlarıyla aynı dizinde olması zorunludur.*

### 3. `3_Model_Egitimi_Python`
Yapay zeka ağırlıklarının (Hex dosyaları) nasıl elde edildiğini kanıtlayan Python eğitim kodunu barındırır.
- `train_ecg_model.py`: Sentetik EKG verisi üretir, modeli eğitir (Gradient Descent) ve Edge-AI donanımına uyumlu olması için tüm sayıları 4-bit'e sıkıştırıp (Quantization) `.hex` çıktıları verir.

## Simülasyon Nasıl Çalıştırılır?
1. Icarus Verilog'da yeni bir proje açın.
2. `2_Simulasyon` klasöründeki tüm `.v` dosyalarını projeye dahil edin (Testbench ve RTL dosyaları).
3. Klasörde yer alan `.hex` dosyalarının derleme dizininde bulunduğundan emin olun.
4. `tb_full_brute_force.v` dosyasını `Top Level Entity` (UUT) olarak seçip simülasyonu başlatın.
5. Dalga formlarında 47. mikrosaniyede `done` sinyalinin 1 olduğunu ve `predicted_class` pininden sonucun (00, 01 veya 10) çıktığını gözlemleyin.

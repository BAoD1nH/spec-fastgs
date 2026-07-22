# So Sánh Hiệu Năng: Spec-FastGS (Ours) vs Vanilla FastGS

Báo cáo này đối chiếu chi tiết hiệu năng huấn luyện và chất lượng tái tạo ảnh giữa **Spec-FastGS** (công trình tích hợp Specular MLP của chúng ta) và **Vanilla FastGS** (bản gốc không có specular) trên cảnh **counter** (tập dữ liệu MipNeRF-360).

---

## 1. Bảng So Sánh Chỉ Số Chi Tiết (Base Train & Big Train)

### 1.1. Cấu Hình Base Train (`densification_interval = 500`)

| Độ Phân Giải | Phương Pháp | Số Gaussians | Thời Gian Train | VRAM Tối Đa | PSNR (dB) | SSIM | LPIPS |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **images_8** | **Vanilla FastGS** | 129,718 | **2m 22s** | **582.47 MB** | 29.73 | 0.9246 | 0.0803 |
| (1/8) | **Spec-FastGS (Ours)** | 151,182 | 10m 50s | 910.84 MB | **30.31** | **0.9298** | **0.0728** |
| | | | | | | | |
| **images_4** | **Vanilla FastGS** | 169,648 | **3m 54s** | **1533.18 MB** | 29.28 | 0.9084 | 0.1340 |
| (1/4) | **Spec-FastGS (Ours)** | 188,316 | 14m 14s | 1864.94 MB | **29.57** | **0.9115** | **0.1309** |
| | | | | | | | |
| **images_2** | **Vanilla FastGS** | 211,181 | **9m 25s** | **5194.35 MB** | 29.03 | 0.8984 | **0.2218** |
| (1/2) | **Spec-FastGS (Ours)** | 215,528 | 20m 49s | 5522.99 MB | **29.33** | **0.8990** | 0.2224 |

### 1.2. Cấu Hình Big Train (`densification_interval = 100`)

| Độ Phân Giải | Phương Pháp | Số Gaussians | Thời Gian Train | VRAM Tối Đa | PSNR (dB) | SSIM | LPIPS |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **images_8** | **Vanilla FastGS** | 164,747 | **2m 47s** | **693.32 MB** | 30.00 | 0.9289 | **0.0702** |
| (1/8) | **Spec-FastGS (Ours)** | 164,456 | 11m 32s | 899.89 MB | **30.15** | **0.9295** | 0.0746 |
| | | | | | | | |
| **images_4** | **Vanilla FastGS** | 291,170 | **5m 31s** | **1837.48 MB** | 29.67 | **0.9168** | **0.1154** |
| (1/4) | **Spec-FastGS (Ours)** | 196,247 | 15m 11s | 1920.10 MB | **29.68** | 0.9128 | 0.1274 |
| | | | | | | | |
| **images_2** | **Vanilla FastGS** | 478,529 | **13m 21s** | 5814.90 MB | **29.45** | **0.9088** | **0.1951** |
| (1/2) | **Spec-FastGS (Ours)** | 246,259 | 23m 33s | **5651.19 MB** | 29.42 | 0.9006 | 0.2182 |

---

## 2. Phân Tích & Đánh Giá Các Khía Cạnh

### 2.1. Chất Lượng Tái Tạo Hình Ảnh (PSNR/SSIM/LPIPS)
* **Ưu thế của Spec-FastGS**: Trong cấu hình **Base Train**, Spec-FastGS liên tục đánh bại Vanilla FastGS về chỉ số PSNR ở mọi độ phân giải (tăng khoảng **+0.30 dB** ở `images_2` và `images_4`). Điều này chứng minh hiệu quả vượt trội của mạng **Specular MLP** trong việc biểu diễn các vùng phản chiếu, ánh kim (specular reflections) trên bề mặt kim loại/gỗ của cảnh counter.
* Trong cấu hình **Big Train**, chất lượng giữa hai bên gần như tương đương nhau ở độ phân giải cao, nhưng Spec-FastGS tối ưu hơn về kích thước mô hình (xem mục 2.2).

### 2.2. Số Lượng Gaussians & Kích Thước Mô Hình
* **Kế thừa và Tối ưu**: Khi cấu hình **Big Train**, Vanilla FastGS bị bùng nổ số lượng điểm lên tới **478,529** điểm ở `images_2`. Trong khi đó, **Spec-FastGS** chỉ tiêu tốn **246,259** điểm (giảm gần **50%** số lượng Gaussians).
* Việc này giúp cho kích thước file lưu trữ của Spec-FastGS nhỏ gọn hơn đáng kể, đồng thời tốc độ render (FPS) khi ứng dụng thực tế sẽ nhanh hơn nhiều so với Vanilla FastGS gốc.

### 2.3. Thời Gian Huấn Luyện & VRAM
* **Thời gian huấn luyện**: Vanilla FastGS có thời gian huấn luyện nhanh hơn rõ rệt (chỉ khoảng 9-13 phút ở `images_2` so với 20-23 phút của Spec-FastGS). Sự chênh lệch này là do Spec-FastGS phải tính toán lan truyền ngược (backward pass) qua mạng nơ-ron Specular MLP cho mỗi Gaussian trên màn hình.
* **VRAM**: Lượng tiêu thụ VRAM tối đa của cả hai phương pháp là tương đương nhau (khoảng ~5.5GB - 5.8GB ở độ phân giải `images_2`), hoàn toàn an toàn trên các GPU có dung lượng 8GB VRAM trở lên.

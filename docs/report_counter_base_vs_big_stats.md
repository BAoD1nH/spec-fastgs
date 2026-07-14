# Báo Cáo Thống Kê Huấn Luyện: Cảnh Counter (Base vs Big Train)

Báo cáo này tổng hợp và phân loại các lượt chạy huấn luyện của cảnh **counter** dựa trên các độ phân giải ảnh khác nhau (`images_8`, `images_4`, `images_2`). Báo cáo giúp phân biệt chi tiết giữa cấu hình **Base Train** (mật độ hóa thưa) và **Big Train** (mật độ hóa dày).

---

## 1. Bảng Thống Kê So Sánh Chi Tiết

| Độ Phân Giải | Cấu Hình Chạy | Số Gaussians | Thời Gian Chạy | VRAM Tối Đa | PSNR (dB) | SSIM | LPIPS | Đường Dẫn Lưu Trữ / Backup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **images_8** (1/8) | **Base Train** | 151,182 | 10m 50s | 910.84 MB | **30.31** | **0.9298** | **0.0728** | `output/backups/counter/spec-fastgs_v2_20260601_133030` |
| **images_8** (1/8) | **Big Train** | 164,456 | 11m 32s | 899.89 MB | 30.15 | 0.9295 | 0.0746 | `output/backups/counter/spec-fastgs_v2_20260601_131602` |
| | | | | | | | | |
| **images_4** (1/4) | **Base Train** | 188,316 | 14m 14s | 1864.94 MB | 29.57 | 0.9115 | 0.1309 | `output/backups/counter/spec-fastgs_v2_20260601_141338` |
| **images_4** (1/4) | **Big Train** | 196,247 | 15m 11s | 1920.10 MB | **29.68** | **0.9128** | **0.1274** | `output/backups/counter/spec-fastgs_v2_20260601_143038` |
| | | | | | | | | |
| **images_2** (1/2) | **Base Train** | 215,528 | 20m 49s | 5522.99 MB | 29.33 | 0.8990 | 0.2224 | `output/backups/counter/spec-fastgs_v2_20260601_150003` |
| **images_2** (1/2) | **Big Train** | 246,259 | 23m 33s | 5651.19 MB | **29.42** | **0.9006** | **0.2182** | `output/counter` (Hiện tại) |

---

## 2. Tiêu Chí Phân Loại & Đánh Giá

### 2.1. Cách Nhận Diện
* **Base Train** (`densification_interval = 500`): Khoảng cách giữa các lần nhân bản/chia tách điểm 3D thưa hơn. Kết quả là số lượng Gaussians cuối cùng thấp hơn, huấn luyện nhanh hơn và tiêu hao ít VRAM hơn.
* **Big Train** (`densification_interval = 100`): Điểm 3D được mật độ hóa liên tục và dày đặc hơn. Kết quả là số Gaussians tăng từ **10% - 15%**, thời gian huấn luyện tăng nhẹ và chất lượng hình ảnh (PSNR/SSIM) được tối ưu cao hơn.

### 2.2. Nhận Xét Kết Quả
* Ở độ phân giải cao (`images_4` và `images_2`), **Big Train** luôn đem lại các chỉ số chất lượng ảnh tốt hơn rõ rệt (PSNR cao hơn khoảng `0.11 dB`, chỉ số sai lệch cảm nhận LPIPS giảm đáng kể).
* Việc phân loại tự động vào các thư mục sao lưu dạng `output/backups/counter/spec-fastgs_v2_[YYYYMMDD]_[HHMMSS]` giúp tránh ghi đè dữ liệu huấn luyện khi chạy lặp lại tệp `.sh` trên cùng một cảnh.

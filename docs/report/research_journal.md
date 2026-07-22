# Nhật Ký Nghiên Cứu & Thực Nghiệm: Spec-FastGS

Tài liệu này đóng vai trò là **Research Journal** (Lab Log) duy nhất của dự án. Thay vì tạo nhiều file báo cáo riêng lẻ gây nhiễu loạn thông tin, toàn bộ hành trình nghiên cứu, ý tưởng, kết quả thực nghiệm và kết luận sẽ được ghi nhận nối tiếp (append) vào file này theo thời gian.

---

## 1. Tổng Quan Dự Án (Repository Overview)

*   **Tên dự án:** Spec-FastGS
*   **Mục tiêu:** Tích hợp mô hình Implicit (Specular MLP) vào nền tảng Explicit (3D Gaussian Splatting / FastGS) nhằm nâng cao chất lượng tái tạo vật liệu phản quang (non-Lambertian surfaces) mà vẫn đảm bảo tốc độ kết xuất thời gian thực (real-time rendering).
*   **Nhánh chính hiện tại:** `spec-fastgs_v2.1.1.2` (đã sửa lỗi Learning Rate của SH, lỗi thiếu tham số densification).
*   **Môi trường thực nghiệm chuẩn:** 
    *   Hệ điều hành: Linux
    *   Thực hiện trên cảnh: `counter` (tập dữ liệu MipNeRF-360)
    *   Cấu hình Big Train chuẩn: `densification_interval = 100`, `grad_abs_thresh = 0.0004`, 30k iterations.

---

## 2. Template Ghi Chép Thực Nghiệm Mới (Copy-Paste Template)

*Mỗi khi bạn bắt đầu một thử nghiệm mới, hãy copy đoạn template dưới đây, paste lên đầu phần **3. Nhật Ký Thực Nghiệm** và điền thông tin.*

```markdown
### 📅 [YYYY-MM-DD HH:MM] | Thực Nghiệm: [Tên Thử Nghiệm]
*   **Nhánh Git / Mốc Code:** `tên-nhánh` hoặc `commit-hash`
*   **Mục tiêu:** [Mô tả ngắn gọn mục đích thử nghiệm là gì, giải quyết vấn đề gì?]

#### 💡 Ý Tưởng & Phương Pháp
*   **Mô tả ý tưởng:** [Tại sao lại thử cách này?]
*   **Thay đổi trong mã nguồn:** 
    *   [File A](file:///absolute/path/to/file): Mô tả thay đổi dòng code...
    *   [File B](file:///absolute/path/to/file): Mô tả thay đổi dòng code...
*   **Lệnh chạy thực nghiệm:** `python train.py ...`

#### 📊 Kết Quả Đạt Được
*So sánh trực tiếp với Baseline gần nhất:*

| Phương pháp | Số Gaussians | Thời gian Train | VRAM | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 000,000 | 00m 00s | 0.0 GB | 00.00 | 0.0000 | 0.0000 |
| **Thực nghiệm mới** | 000,000 | 00m 00s | 0.0 GB | 00.00 | 0.0000 | 0.0000 |

*   **Nhận xét định tính (Visual Quality):** [Ảnh có bị mờ, có bị nhiễu floaters không? Vệt sáng phản chiếu có mượt không?]

#### 🏁 Kết Luận & Hướng Đi Tiếp Theo
*   [ ] Đạt yêu cầu / Thất bại? Lý do?
*   [ ] Hướng đi tiếp theo: [Ví dụ: tăng LR, đổi hàm loss...]
```

---

## 3. Nhật Ký Thực Nghiệm (Research Log)

*(Các thực nghiệm mới nhất sẽ được thêm lên đầu phần này)*

### 📅 2026-06-04 13:30 | Thực Nghiệm: Spec-FastGS v2.1.1.2 vs Vanilla FastGS (Big Train)
*   **Nhánh Git / Mốc Code:** `spec-fastgs_v2.1.1.2` (đã fix bug SH Learning Rate `/20.0`, `max_screen_size=20`, `grad_abs_thresh=0.0004`).
*   **Mục tiêu:** Đánh giá hiệu năng và chất lượng của Spec-FastGS trên cả 4 độ phân giải ảnh của cảnh `counter` khi hệ thống densification đã được đưa về đúng quỹ đạo chuẩn.

#### 💡 Ý Tưởng & Phương Pháp
*   **Mô tả ý tưởng:** Sau khi phát hiện ra lỗi nghiêm trọng ở các phiên bản trước (làm bùng nổ số lượng điểm vô hại hoặc kìm hãm số lượng điểm quá mức), ta khôi phục lại cơ chế Densification chuẩn của FastGS. Đồng thời cho phép Specular MLP tối ưu hóa màu sắc phản chiếu bắt đầu từ iteration `15,000` (để SH học hình học trước).
*   **Thay đổi trong mã nguồn:** Vá lỗi LR của SH trong [train.py](file:///home/baodinh/spec-fastgs/train.py) và cấu hình lại file chạy [run_spec-fastgs_big.sh](file:///home/baodinh/spec-fastgs/run_spec-fastgs_big.sh).

#### 📊 Kết Quả Đạt Được
*So sánh trên cấu hình Big Train (30k iterations):*

| Tập ảnh (Scale) | Phương pháp | Số Gaussians | Thời gian Train | VRAM Tối Đa | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **images_8** | Vanilla FastGS | 164,747 | **2m 47s** | **693.32 MB** | 30.00 | 0.9289 | 0.0702 |
| *(Phân giải 1/8)* | **Spec-FastGS (v2.1.1.2)**| 201,957 | 13m 40s | 975.42 MB | **30.13** (+0.13) | **0.9357** | **0.0652** |
| | | | | | | | |
| **images_4** | Vanilla FastGS | 291,170 | **5m 31s** | **1837.48 MB** | 29.67 | 0.9168 | 0.1154 |
| *(Phân giải 1/4)* | **Spec-FastGS (v2.1.1.2)**| 358,472 | 26m 05s | 2322.81 MB | **29.77** (+0.10) | **0.9215** | **0.1098** |
| | | | | | | | |
| **images_2** | Vanilla FastGS | 478,529 | **13m 21s** | **5814.90 MB** | 29.45 | 0.9088 | 0.1951 |
| *(Phân giải 1/2)* | **Spec-FastGS (v2.1.1.2)**| 600,979 | 49m 10s | 6614.17 MB | **29.51** (+0.06) | **0.9119** | **0.1898** |
| | | | | | | | |
| **images** | Vanilla FastGS | 468,793 | **14m 04s** | **6177.02 MB** | 29.56 | 0.9170 | 0.1772 |
| *(Phân giải gốc 1x)*| **Spec-FastGS (v2.1.1.2)**| 592,200 | 50m 08s | 6958.03 MB | **29.57** (+0.01) | **0.9187** | **0.1730** |

#### 🏁 Kết Luận & Hướng Đi Tiếp Theo
*   **Đánh giá:** Thành công. Spec-FastGS cải thiện toàn diện các chỉ số chất lượng ảnh, đặc biệt là LPIPS (giảm tới 7.1% lỗi perceptional). Số lượng Gaussians tăng trưởng ổn định (chỉ nhiều hơn FastGS khoảng 22% - 26%).
*   **Vấn đề tồn tại:** Ở độ phân giải gốc (`images` 1x), mức độ cải thiện PSNR thu hẹp lại đáng kể (+0.01 dB) do mật độ Gaussians quá cao lấn át gradient của MLP, kết hợp việc tắt regularizer `spec_reg`.
*   **Hướng đi tiếp theo:**
    *   [ ] Thực nghiệm **Method 1B** (strategy 2 của parameter tuning) - giảm `grad_abs_thresh` động sau khi bật MLP ở iter 3k.
    *   [ ] Thực nghiệm **Method 2** (Decoupled Densification) - phân tách luồng gradient hình học và màu sắc để tối ưu hóa quá trình sinh điểm.

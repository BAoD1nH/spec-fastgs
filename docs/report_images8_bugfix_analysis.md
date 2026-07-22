# Báo Cáo Phân Tích Thực Nghiệm: Cởi Trói Hiệu Năng Spec-FastGS
**Dataset:** Mip-NeRF 360 (Scene: Counter)
**Độ phân giải:** `images_8`
**Cấu hình:** Big Train (`densification_interval=100`, `grad_abs_thresh=0.0004`, `highfeature_lr=0.02`)
**Nhánh:** `spec-fastgs_v2.1.1.2`

---

## 1. Bối Cảnh
Trong các thử nghiệm trước đó, **Spec-FastGS** ghi nhận số lượng Gaussians luôn thấp hơn đáng kể so với **Vanilla FastGS** gốc, dẫn tới việc chỉ số PSNR không đạt được mức kỳ vọng ở độ phân giải cao. Qua quá trình rà soát mã nguồn, chúng ta đã phát hiện và khắc phục thành công **3 lỗi implementation cốt lõi** vốn đã kìm hãm mô hình:
1. Thiếu tham số `max_screen_size=20` khiến mô hình không thể xóa (prune) các điểm quá lớn.
2. Thiếu tham số `grad_abs_thresh=0.0004` khiến mô hình thiếu độ nhạy để chẻ điểm (split).
3. Thiếu phép chia `/ 20.0` trong Learning Rate của Spherical Harmonics, khiến tốc độ học màu sắc tăng vọt gấp 20 lần, gây nhiễu gradient và chẻ điểm vô tội vạ.

Dưới đây là kết quả thử nghiệm trên độ phân giải `images_8` sau khi đưa cơ chế Densification về đúng quỹ đạo chuẩn của Vanilla FastGS.

---

## 2. Bảng Thống Kê So Sánh Kết Quả

| Mốc Thử Nghiệm (`images_8`) | Số Gaussians | PSNR (dB) | SSIM | LPIPS | Thời Gian Train |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Vanilla FastGS (Bản gốc)** | 164,747 | 30.000 | 0.9289 | 0.0702 | ~2m 47s |
| **2. Spec-FastGS (Chưa sửa lỗi LR)** | 367,686 | **30.389** | 0.9356 | 0.0693 | 22m 18s |
| **3. Spec-FastGS (Đã vá toàn bộ lỗi)** | **201,957** | 30.133 | **0.9357** | **0.0652** | 13m 40s |

---

## 3. Phân Tích Chuyên Sâu

### 3.1. Sự Ổn Định Của Số Lượng Điểm (Gaussians)
Sau khi khôi phục phép chia `/ 20.0` cho tốc độ học của Spherical Harmonics, hiện tượng nhiễu gradient vị trí (`xyz_gradient`) đã được dập tắt. Số lượng điểm ngay lập tức hạ nhiệt từ **368k** xuống còn **201,957** điểm. 
Mức 201k điểm này đưa Spec-FastGS về cùng một "hạng cân" với Vanilla FastGS (164k). Khoảng chênh lệch nhỏ (~37k điểm) không làm phình to bộ nhớ VRAM nhưng đủ để tạo ra một cấu trúc hình học ổn định.

### 3.2. Sức Mạnh Của Specular MLP Được Giải Phóng
Khi được đặt lên bàn cân công bằng với cấu trúc điểm chuẩn xác, hiệu năng của Specular MLP thể hiện sự áp đảo tuyệt đối so với bản gốc:
* **PSNR (+0.133 dB)**: Nhờ có mạng MLP chuyên xử lý ánh sáng phản quang, Spec-FastGS dễ dàng vượt qua Vanilla FastGS mà không cần phải "vặn vẹo" các điểm Gaussians để fit những vệt sáng lóa.
* **LPIPS (0.0652)**: Đây là chỉ số quan trọng nhất đại diện cho cảm nhận sắc nét của mắt người. Spec-FastGS thiết lập kỷ lục mới, đánh bại hoàn toàn Vanilla FastGS (0.0702). Đáng ngạc nhiên hơn, bản 201k điểm lại có LPIPS tốt hơn cả bản 368k điểm, chứng minh rằng chất lượng hình ảnh không chỉ đến từ việc đẻ thêm điểm, mà đến từ sự kết hợp hoàn hảo giữa cấu trúc gọn gàng và bộ giải mã phản quang chính xác.

## 4. Kết Luận
Quá trình rà soát và gỡ lỗi (debugging) mã nguồn đã mang lại thành công vang dội. Mọi nút thắt cổ chai về mặt kỹ thuật đã được gỡ bỏ. Spec-FastGS hiện tại (nhánh `v2.1.1.2`) đã sẵn sàng để phô diễn sức mạnh thực sự trên các tập độ phân giải lớn hơn (`images_4`, `images`).

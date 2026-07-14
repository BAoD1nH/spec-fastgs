# Đề xuất Cải tiến Kiến trúc spec-fastgs

Dựa trên việc bóc tách 6 điểm yếu của kiến trúc hiện tại và đối chiếu với các "selling points" từ 5 bài báo SOTA (ARS-GS, Aniso-GS, MaterialRefGS, RGS-DR, AuGS), dưới đây là các đề xuất chiến lược để nâng cấp `spec-fastgs`. Mục tiêu là đạt được sự cân bằng tối ưu: **Tốc độ render siêu nhanh (Fast)** và **Chất lượng phản xạ xuất sắc (Quality)**.

Các đề xuất được chia thành 2 nhóm ưu tiên để bạn dễ dàng lựa chọn định hướng phát triển tiếp theo.

---

## NHÓM 1: ĐỘT PHÁ VỀ TỐC ĐỘ (OPTIMIZATION UPGRADES)
Nhóm này giải quyết trực tiếp nút thắt cổ chai (bottleneck) về FPS và VRAM của hệ thống hiện tại, đồng thời vô tình giải quyết luôn hiện tượng nhòe phản xạ.

### 1. Deferred Specular Shading (Lấy cảm hứng từ RGS-DR)
*   **Giải quyết điểm yếu:** Nhòe phản xạ do Alpha-blending (Điểm yếu 3) & FPS thấp do MLP chạy trên 3D.
*   **Giải pháp:** 
    *   Sửa đổi `gaussian_renderer` để nó không xuất ra ảnh RGB cuối cùng ngay lập tức. Thay vào đó, nó xuất ra các bức ảnh 2D trung gian (G-Buffer) chứa: Màu Diffuse, Tọa độ bề mặt, và Đặc trưng ẩn (Latent Specular Features).
    *   Đẩy mạng `SpecularNetwork` ra khỏi quá trình kết xuất 3D. Mạng này sẽ chạy dưới dạng mạng CNN 2D hoặc MLP tính toán trực tiếp trên từng pixel của ảnh G-Buffer 2D.
*   **Hiệu quả mong đợi:** 
    *   **Tốc độ:** FPS sẽ tăng vọt vì phép tính nặng nhất (MLP) giờ chỉ chạy trên độ phân giải màn hình ($W \times H$), không phụ thuộc vào việc cảnh có 1 triệu hay 10 triệu Gaussians.
    *   **Visual:** Các đốm sáng bóng loáng (highlights) sẽ sắc như dao cạo, vì chúng được vẽ trực tiếp lên pixel cuối cùng, hoàn toàn thoát khỏi sự bôi nhòe của Alpha-blending.

### 2. Frequency/Variance-based Specular Masking (Lấy cảm hứng từ Aniso-GS & MaterialRefGS)
*   **Giải quyết điểm yếu:** Mạng tự đoán mò gây sinh rác (Điểm yếu 6) & Lãng phí tài nguyên tính toán.
*   **Giải pháp:** 
    *   Tính toán "Reflection Strength Prior" (Phương sai màu sắc qua nhiều góc nhìn) hoặc thống kê Tần suất xuất hiện của Gaussian.
    *   Tạo một cơ chế **Công tắc (Switch)**: Chỉ kích hoạt nhánh tính toán Specular cho những điểm Gaussian thuộc vùng có phương sai cao (kim loại, kính). Các vùng tường, thảm (phương sai thấp) sẽ bị tắt hoàn toàn nhánh Specular.
*   **Hiệu quả mong đợi:** Tiết kiệm một lượng khổng lồ VRAM và phép toán vô ích. Mô hình trở nên cực kỳ sạch, không còn các đám mây rác (floaters) phát sáng lơ lửng.

---

## NHÓM 2: ĐỘT PHÁ VỀ CHẤT LƯỢNG (PHOTOMETRIC & GEOMETRY UPGRADES)
Nhóm này tập trung vào việc làm cho đốm sáng chân thực hơn và sửa lỗi rách bề mặt hình học.

### 3. Spatially Adaptive Features - SASHF (Lấy cảm hứng từ Aniso-GS)
*   **Giải quyết điểm yếu:** Đầu vào MLP quá "tù mù", khó bắt chi tiết cao (Điểm yếu 1).
*   **Giải pháp:** 
    *   Thay vì truyền trực tiếp tọa độ $xyz$ hay hướng nhìn vào SpecularNetwork, hãy nhúng tọa độ bằng một **Multi-resolution Hash Grid** (cực nhanh, lấy từ Instant-NGP), sau đó nhân Tensor với vector hướng nhìn.
*   **Hiệu quả mong đợi:** Hash Grid cung cấp nhận thức không gian cực mạnh. Mạng `SpecularNetwork` (dù rất nông và nhỏ) vẫn có thể dự đoán chính xác các tia sáng chớp lóa (glints) nhỏ xíu trên bề mặt cong.

### 4. 2D-to-3D Inverse Splatting cho Densification (Lấy cảm hứng từ AuGS)
*   **Giải quyết điểm yếu:** Gradient bị triệt tiêu làm rách lưới (Điểm yếu 5) & Nảy nở sai vị trí trong 3D.
*   **Giải pháp:** 
    *   Nâng cấp cơ chế `Error-Driven Densification` hiện tại (Hướng A). Khi phát hiện vùng có Specular Error lớn trên **ảnh render 2D**, ta tạo các điểm Gaussian sửa lỗi ngay trên màn hình 2D đó.
    *   Sử dụng Depth Map để "bắn ngược" (Inverse Project) tọa độ 2D này thành tọa độ 3D, và thêm chúng vào danh sách Gaussians của cảnh.
*   **Hiệu quả mong đợi:** Trực tiếp vượt qua rào cản Vanishing Gradient. Đảm bảo các hạt Gaussian sinh ra để sửa lỗi ánh sáng sẽ dán dính hoàn hảo vào bề mặt vật thể, che lấp các lỗ hổng hình học mà không sinh rác.

### 5. View-dependent Opacity Lobe (Lấy cảm hứng từ AuGS)
*   **Giải quyết điểm yếu:** Phản xạ bị lem màu.
*   **Giải pháp:** Thay vì bắt mạng đoán màu Specular, hãy tạo một lớp Gaussian thứ hai đè lên bề mặt. Lớp này có màu sáng trắng nhưng có **độ đục (Opacity) thay đổi theo góc nhìn**. Nó tàng hình khi nhìn sai góc và hiện rõ khi đúng góc phản xạ.
*   **Hiệu quả mong đợi:** Tách bạch (Disentangle) hoàn toàn màu cơ bản và ánh sáng lấp lánh. Khả năng Edit/Relight trong tương lai sẽ vô cùng dễ dàng.

---

## TỔNG KẾT & GỢI Ý LỘ TRÌNH (ROADMAP)

Nếu bắt tay vào code phiên bản mới (giả sử gọi là **Hướng B**), lộ trình thực thi mang lại giá trị cao nhất so với công sức (High ROI) sẽ là:

1.  **Bước 1 (Dọn dẹp rác & Tăng tốc độ):** Áp dụng **Giải pháp 2** (Variance-based Masking) làm trọng số phụ trợ cho Error-Driven Densification hiện tại. Code rất dễ, hiệu quả thấy ngay.
2.  **Bước 2 (Chất lượng ánh sáng):** Áp dụng **Giải pháp 3** (Hash Grid SASHF). Tích hợp thư viện `tiny-cuda-nn` để nhúng Hash Grid làm đầu vào cho Specular MLP. Đốm sáng sẽ sắc nét hơn.
3.  **Bước 3 (Đại phẫu Kiến trúc - Ultimate Upgrade):** Thực hiện **Giải pháp 1** (Deferred Specular Shading). Bước này đòi hỏi can thiệp sâu vào file `.cu` của `gaussian_renderer` để xuất G-Buffer, nhưng phần thưởng là FPS sẽ tăng gấp nhiều lần, biến `spec-fastgs` thành một hệ thống "Fast" đích thực theo chuẩn SOTA.

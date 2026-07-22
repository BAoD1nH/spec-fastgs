# Cơ chế tương tác giữa ASG và Ref Score trong 1 Iteration

Câu hỏi của bạn chạm đến đúng cốt lõi của kiến trúc Neural Gating (Cổng Neural) trong nhánh `spec-fastgs-unofficial`. Tài liệu này sẽ bóc tách chi tiết chính xác những gì diễn ra ở **một vòng lặp duy nhất (1 iteration)** trong quá trình training.

Đầu tiên, cần khẳng định: **Không phải tất cả camera views đều được tính toán trong 1 iteration.**
Giống như mọi thuật toán SGD (Stochastic Gradient Descent) khác của học sâu, ở mỗi vòng lặp, hệ thống chỉ bốc ngẫu nhiên **DUY NHẤT 1 BỨC ẢNH (1 Camera View)** từ tập dữ liệu để xử lý. Việc xử lý tất cả camera cùng lúc sẽ làm tràn bộ nhớ VRAM ngay lập tức.

Dưới đây là chu trình chi tiết tại **1 vòng lặp (Iteration thứ K)** với camera góc nhìn $C$:

## Bước 1: Mạng ASG dự đoán màu phản xạ
1. Hệ thống xác định xem trong số hàng triệu hạt 3D Gaussians, những hạt nào đang **nằm trong tầm nhìn** của camera $C$. Giả sử có $M$ hạt hiển thị.
2. Mạng ASG sẽ nhận đầu vào là: Hướng nhìn (`viewdir`), pháp tuyến (`normal`), và đặc trưng riêng của hạt (`f_asg`).
3. Mạng ASG tính toán và "phun" ra một mảng màu Specular: `spec_sparse` (Kích thước: $M \times 3$). Đây là lượng ánh sáng lấp lánh mà mạng ASG *đoán* là sẽ có.

## Bước 2: Ref Score thực hiện Gating (Nhân với ASG)
1. Camera $C$ đã có sẵn một bản đồ **Ref Score 2D** (được tính toán offline từ trước).
2. Hệ thống lấy tọa độ 3D (`xyz`) của $M$ hạt Gaussians kia, **chiếu ngược** nó lên tọa độ pixel 2D của bức ảnh Ref Score này.
3. Tại mỗi điểm pixel rơi trúng, hệ thống "đọc" lấy giá trị Ref Score (từ 0.0 đến 1.0). Ta thu được mảng `sampled_ref_score` (Kích thước $M$).
4. **THỰC HIỆN NHÂN (GATING)**: `spec_sparse_gated = spec_sparse * sampled_ref_score`.
   - Hạt nào chiếu trúng pixel có Ref Score = 0 (vùng nhám): Ánh sáng ASG bị dập tắt về 0.
   - Hạt nào chiếu trúng pixel có Ref Score = 1 (vùng bóng): Ánh sáng ASG được giữ nguyên.

## Bước 3: Render (Kết xuất) và Tính Loss
1. Màu sắc cuối cùng của từng hạt Gaussian lúc này được cộng gộp lại: $C_{final} = C_{diffuse} + \text{spec\_sparse\_gated}$
2. Trình Rasterizer của FastGS sẽ trộn tất cả các hạt này lại (dựa trên Opacity và Alpha) để tạo ra bức ảnh render 2D.
3. Lấy bức ảnh render này trừ đi ảnh thực tế (Ground Truth) của camera $C$ để tính ra sai số **Loss (L1 + SSIM)**.

## Bước 4: Backpropagation - Quay lại cập nhật cái gì?
Khi gọi lệnh `loss.backward()`, sai số sẽ chạy ngược từ bức ảnh, đi qua Rasterizer, rồi tách làm 2 nhánh để cập nhật trọng số. Đây là lúc phép nhân Ref Score phát huy tác dụng khổng lồ:

Vì Loss đạo hàm đi qua phép nhân `spec_sparse * ref_score`, nên **Gradient truyền về mạng ASG sẽ bị nhân với Ref Score**.
- **Nếu Ref Score = 0**: Lượng đạo hàm truyền về mạng ASG = 0. Mạng ASG không hề bị phạt hay thay đổi gì cả, nó **KHÔNG HỌC** gì ở vùng này. Toàn bộ trọng trách sửa sai được đẩy hết cho Base (Diffuse).
- **Nếu Ref Score = 1**: Đạo hàm nguyên vẹn truyền về mạng ASG. ASG phải nhận trách nhiệm cập nhật trọng số để khớp với đốm sáng.

**Hệ thống sẽ cập nhật (Optimizer Step) cho 2 nhóm chính:**
1. **Model 3D Gaussians**: Hệ số khuếch tán (Diffuse - `f_dc`), tọa độ 3D (`xyz`), độ xoay (`rotation`), kích thước (`scaling`), độ mờ (`opacity`), và đặc trưng riêng của ASG (`f_asg`).
2. **Mạng MLP của ASG**: Trọng số (weights) của các lớp Linear Layers bên trong mạng Neural.

> [!TIP]
> **Tóm tắt ý nghĩa:**
> Phép nhân Ref Score như một "công tắc" dòng chảy đạo hàm. Nó bảo mạng ASG rằng: *"Chỗ này là gỗ nhám (Ref=0), sai số ở đây không phải lỗi của mày, mày không cần cập nhật hệ số. Nhưng chỗ kia là sắt bóng (Ref=1), sai số ở đó là do mày đoán ánh sáng sai, hãy tự cập nhật trọng số (weights) của mày đi!"*

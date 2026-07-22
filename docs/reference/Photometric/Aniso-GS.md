# Phân tích Aniso-GS (Anisotropic appearance field for complex highlight modeling in 3D Gaussian splatting)

## 1. Selling Points của Aniso-GS
Paper **Aniso-GS** tập trung vào việc giải quyết nhược điểm của Spherical Harmonics (SH) bậc thấp trong việc mô hình hóa các chi tiết tần số cao (đặc biệt là ánh sáng phản xạ sắc nét - specular highlights). Thay vì dùng một mô hình vật lý phức tạp (như BRDF hay ASG), Aniso-GS sử dụng cách tiếp cận **Feature Engineering** và **Training Strategy**. Các selling points chính bao gồm:

1. **Spatially Adaptive Spherical Harmonic Features (SASHF)**:
   - Họ nhận thấy tăng bậc SH không cải thiện nhiều nhưng làm tăng kích thước mô hình, còn ASG thì phức tạp. 
   - Giải pháp: Mã hóa vị trí `xyz` bằng một **Multi-resolution Hash Grid** để lấy ra feature $u$ (32 chiều). Mã hóa hướng nhìn bằng SH để lấy feature $d$ (16 chiều). Sau đó dùng phép nhân Tensor (Tensor Product $u \otimes d$) để tạo ra một feature không gian nhiều chiều (512 chiều). Việc phóng các SH bậc thấp lên không gian chiều cao giúp mô hình dễ dàng học được các chi tiết chói sáng tần số cao.

2. **Frequency-Driven Adaptive Activation Strategy**:
   - Nếu áp dụng SASHF cho mọi 3D Gaussians, mô hình sẽ bị overfitting ở các Gaussians ít xuất hiện (low-frequency observation - bị che khuất hoặc chỉ thấy ở 1-2 góc).
   - Aniso-GS thống kê tần suất xuất hiện (occurrence frequency) của từng Gaussian. Nếu tần suất cao (vùng có view-dependency mạnh), dùng SASHF. Nếu thấp, chỉ dùng phép nối (concatenation $u \oplus d$) thông thường.

3. **Multi-View Information Smoothing (M-Smoothing)**:
   - Vì 3DGS train từng ảnh đơn (single-view optimization), mô hình dễ bị cuốn vào local optimum (overfit cho những ảnh cuối cùng).
   - Ở các epoch cuối, họ áp dụng Exponential Moving Average (EMA) để làm mịn (smooth) trọng số mô hình qua các views, giúp giữ được sự nhất quán (consistency) và tăng khả năng tổng quát hóa (generalization) khi render ở góc nhìn mới.

4. **Adaptive Densification Strategy**:
   - Thay vì dùng một ngưỡng (threshold) cố định để clone/split Gaussians, họ dùng một chiến lược tích lũy gradient vị trí (view-space position gradients) có trọng số dựa trên tần suất xuất hiện. 
   - Giúp kìm hãm sự phình to của Gaussians ở các vùng ít quan sát (rác, floaters) và thúc đẩy tạo Gaussians mới ở các vùng quan trọng.

---

## 2. Ý tưởng có thể tận dụng cho `spec-fastgs`

Với định hướng "Cân bằng giữa Photometric và Optimization", các kỹ thuật của Aniso-GS rất phù hợp vì chúng cải thiện chất lượng mà **không dùng các mô hình quang học nặng nề**.

### A. Spatially Adaptive SH Features (SASHF) kết hợp SpecularNetwork
- **Áp dụng:** Thay vì đưa trực tiếp hướng nhìn (view direction) hoặc SH vào `SpecularNetwork` (MLP), bạn có thể dùng **Multi-resolution Hash Grid** để trích xuất vị trí, sau đó nhân Tensor với View Direction để tạo input cho MLP.
- **Lợi ích:** Tensor Product cung cấp một lượng thông tin không gian (spatial) cực mạnh, giúp nhánh Specular MLP học các đốm sáng phản xạ sắc nét (sharp highlights) dễ dàng hơn nhiều mà không cần làm MLP quá sâu. Hash Grid tính toán cực kỳ nhanh (như trong Instant-NGP), nên rất tối ưu về mặt Optimization.

### B. Frequency-Driven Control (Điều khiển bằng tần suất quan sát)
- **Áp dụng:** Hiện tại "Hướng A" của `spec-fastgs` dùng Specular Error để kích hoạt Densification. Ta có thể bổ sung thêm **hệ số tần suất quan sát (Occurrence Frequency)**. Các Gaussian nào chỉ xuất hiện ở < 5% số lượng camera thì giảm ngưỡng Densification hoặc **tắt hẳn Specular Branch** của chúng.
- **Lợi ích:** Tiết kiệm một lượng lớn phép tính MLP và RAM cho những điểm ảnh không quan trọng hoặc điểm nhiễu (floaters), dồn tài nguyên cho các bề mặt phản xạ chính.

### C. Multi-View Information Smoothing (M-Smoothing)
- **Áp dụng:** Ở khoảng 5000 iterations cuối cùng (trước khi kết thúc training), kích hoạt EMA để tính trung bình các trọng số (nhất là trọng số của `SpecularNetwork` và vị trí Gaussian).
- **Lợi ích:** Giảm hiện tượng ánh sáng lấp lánh (flickering) hoặc bị sai màu khi di chuyển camera (view synthesis), cải thiện trực tiếp các metrics như SSIM và LPIPS.

---

## 3. So sánh Trước và Sau khi tích hợp (Before / After)

| Tiêu chí | Trước khi áp dụng (Current `spec-fastgs`) | Sau khi áp dụng ý tưởng từ Aniso-GS |
| :--- | :--- | :--- |
| **Độ sắc nét của Specular (Photometric)** | Các đốm sáng phản chiếu (highlights) có thể bị mờ hoặc không tụ lại thành chi tiết sắc nét do giới hạn của SH và MLP. | Khả năng bắt chi tiết tần số cao cực tốt nhờ không gian chiều cao của **SASHF (Tensor Product)**. Các đốm sáng bóng loáng như kim loại, kính sẽ rõ ràng hơn. |
| **Tối ưu số lượng Gaussians (Optimization)** | Các Gaussian không quan trọng (floaters, phông nền bị che) vẫn có thể bị tính toán phản xạ và sinh sôi nảy nở (densify) sai lầm. | Số lượng Gaussian được kiểm soát khắt khe hơn. Vùng góc khuất bị **chặn densify** (nhờ Adaptive Densification) và **tắt MLP** (nhờ Adaptive Activation). Memory cực kỳ tối ưu. |
| **Sự ổn định giữa các khung hình (View-Consistency)** | Khi render video nội suy (novel views), đôi khi màu sắc hoặc phản xạ có thể bị nhảy/chớp (flickering) nhẹ do overfit từng ảnh. | Render mượt mà hơn, màu sắc phản xạ chuyển động trơn tru nhờ **Multi-View Information Smoothing** ở giai đoạn cuối. |

**Kết luận:**
Khác với ARS-GS mang nặng tính vật lý quang học (PBR), Aniso-GS thuần túy là **kỹ thuật về Data & Features**. Cả hai đều có thể áp dụng cho `spec-fastgs`:
- Nếu muốn hướng bề mặt phản xạ chính xác tuyệt đối như thực tế $\rightarrow$ Ưu tiên hàm $G_{ASG}$ và Skip Connection của **ARS-GS**.
- Nếu muốn một giải pháp cực kỳ "Fast" (nhanh), cấu trúc dữ liệu gọn gàng và bắt Highlight sắc bén $\rightarrow$ Ưu tiên **SASHF (Hash Grid + Tensor Product)** và **Adaptive Control** của **Aniso-GS**. Tích hợp cái này có thể dễ lập trình (coding) hơn so với việc tính toán PBR.

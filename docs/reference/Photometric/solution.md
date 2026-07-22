Listed directory reference
Listed directory docs
Ran command: `find /home/baodinh/spec-fastgs -name "*.md" | grep -i photometric | head -20 && echo "---" && find /home/baodinh/spec-fastgs -type d | head -20`
Used tool: ask_permission
Listed directory baodinh
Listed directory baodinh_thesis
Listed directory spec-fastgs
Listed directory docs
Listed directory reference
Listed directory Photometric

Tốt, tôi sẽ đọc tất cả các file phân tích `.md` để hiểu rõ bức tranh tổng thể trước khi đưa ra khuyến nghị.

Viewed GaussianShader.md:1-65
Viewed 3DGS-DR.md:1-110
Viewed ReflectiveGS.md:1-43
Viewed RGS-DR.md:1-112
Viewed MaterialRefGS_analysis.md:1-155
Viewed AuGS.md:1-156

Tôi đã đọc xong toàn bộ 5 bài phân tích. Bây giờ hãy để tôi tổng hợp bức tranh tổng thể và đưa ra lời khuyên chiến lược thẳng thắn cho bạn.

---

## Phân tích toàn cảnh: `spec-fastgs` đang ở đâu và nên đi đâu?

### I. Vấn đề cốt lõi bạn đang gặp phải

Bạn đặt câu hỏi rất đúng: **"Balance không biết cải tiến cái nào?"**

Lý do bạn cảm thấy bị kẹt là vì **`spec-fastgs` hiện tại chưa có một "đặc trưng độc đáo" (Unique Selling Point) rõ ràng**. FastGS đã nhanh nhưng mất specular. Specular-Gaussian đã có specular nhưng chậm. Việc ghép lại 2 cái này cho ra mô hình "nhanh hơn SpecGaussian, tốt hơn FastGS" — nghe hay nhưng chưa đủ "wow" cho reviewer vì đây là kỹ thuật "kết hợp đơn thuần" (naive combination).

Đây là lý do tại sao bạn không biết ưu tiên Photometric hay Optimization:
> **Bạn đang cố gắng cải thiện một mô hình chưa có Architecture Foundation đủ mạnh để cải tiến thêm.**

---

### II. Thông điệp nhất quán từ 5 bài phân tích

Đọc xong cả 5 file, tôi thấy **tất cả các paper đều hội tụ về một điểm chung**, và đây là điều `spec-fastgs` hiện tại **hoàn toàn thiếu**:

| Vấn đề | Spec-FastGS hiện tại | SOTA (2024-2026) |
|---|---|---|
| **Shading** | Per-Gaussian, ASG + MLP đoán màu ngầm định | Deferred (per-pixel): G-Buffer → BRDF |
| **Normal** | Không có loss, chỉ là input | Có Normal Consistency Loss, explicit từ 2DGS |
| **Material** | Không có (ASG học "mờ" tất cả) | Explicit: Albedo + Roughness + Specular Tint |
| **Densification** | Gradient ngẫu nhiên, cả specular & diffuse lẫn lộn | Error-driven: ưu tiên vùng specular lỗi cao |
| **Cross-view** | Không ràng buộc | Multi-view Consistency Loss |

---

### III. Lời khuyên chiến lược: Có nên tiếp tục dựa trên FastGS không?

Câu trả lời là **Có, nhưng phải thay đổi câu chuyện (Narrative)**.

Thay vì cố gắng cải tiến dàn trải, hãy **chọn một trong 2 hướng đi rõ ràng**:

---

#### 🔴 Hướng A: "Fast + Good Enough" — Vẫn giữ FastGS, cải thiện nhẹ về Photometric

**Mục tiêu:** Chứng minh rằng một số kỹ thuật Photometric rẻ (chi phí thấp) có thể cải thiện chất lượng specular mà **không đánh đổi tốc độ**.

**Các cải tiến cụ thể (sắp xếp theo dễ → khó):**
1. **Bật lại `spec_reg` + thêm `L_normal`** (GaussianShader) — ~15 dòng code, không đổi architecture. Dự kiến +0.5–1.0 dB PSNR.
2. **Error-Driven Densification** (AuGS) — thêm 1 phase densification có định hướng specular vào `train.py`. Dự kiến +0.5–0.7 dB PSNR tại vùng specular.
3. **Reflection Strength Prior** (MaterialRefGS) — tạo per-pixel weight map từ photometric variance để guide ASG loss. Thấp-trung bình complexity.

**Metrics cải thiện:** PSNR, SSIM, LPIPS ↑ | FPS gần như không đổi | Training time tăng nhẹ.

**Narrative bài báo/khóa luận:** *"Chúng tôi chỉ ra rằng các kỹ thuật photometric có chi phí thấp (low-cost normal supervision + specular-aware densification) có thể cải thiện đáng kể chất lượng bề mặt phản xạ trong framework tốc độ cao mà không đánh đổi FPS."*

---

#### 🔵 Hướng B: "Abandon FastGS, Build Proper Architecture" — Đây là hướng cho Paper

**Nếu mục tiêu là publish paper quốc tế**, bạn nên nghiêm túc xem xét **không dùng FastGS làm base nữa**, mà build từ một trong:
- **2DGS** (2D Gaussian Splatting) — có normal tốt hơn từ đầu, phù hợp với Deferred Shading
- **3DGS-MCMC** — kiểm soát số lượng Gaussian tốt hơn, kết hợp tốt với AuGS-style enhancement

Rồi tích hợp theo pipeline của **3DGS-DR hoặc RGS-DR** (Deferred Shading + G-Buffer) + nhánh FastGS-style visibility masking để giữ tốc độ.

**Lý do:** Tất cả SOTA mới nhất (ReflectiveGS: 122 FPS, RGS-DR, MaterialRefGS) đều dùng 2DGS + Deferred, vẫn đạt **real-time**. Tức là bạn **không cần FastGS để đạt tốc độ** — kiến trúc 2DGS + Deferred đã đủ nhanh.

---

### IV. Kết luận thực tế

| Mục tiêu | Khuyến nghị |
|---|---|
| **Bảo vệ khóa luận (ngắn hạn)** | Chọn **Hướng A** — Giữ FastGS, thêm `L_normal` + Error-Driven Densification. Dễ làm, kết quả rõ ràng, đủ để viết thesis. |
| **Publish paper (dài hạn)** | Chọn **Hướng B** — Rebuild từ 2DGS + Deferred Shading, dùng FastGS chỉ như reference baseline để so sánh tốc độ. |

**Gợi ý thực tế nhất ngay lúc này:** Nếu deadline khóa luận gần, hãy đi **Hướng A** với 2 việc cụ thể:
1. Thêm `L_normal` (Normal Consistency Loss) vào `train.py` — tôi có thể làm điều này ngay cho bạn.
2. Implement Error-Driven Densification từ AuGS — tập trung densify vùng specular lỗi cao.

Bạn muốn tôi bắt tay vào cài đặt cụ thể **Hướng A** ngay bây giờ không?
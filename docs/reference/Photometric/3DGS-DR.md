Listed directory Photometric
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2404.18454v2_3DGS-DR.pdf - 2>/dev/null | head -n 400`
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2404.18454v2_3DGS-DR.pdf - 2>/dev/null | sed -n '400,800p'`
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2404.18454v2_3DGS-DR.pdf - 2>/dev/null | sed -n '800,1100p'`
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2404.18454v2_3DGS-DR.pdf - 2>/dev/null | tail -n 200`

Tôi đã đọc toàn bộ nội dung của paper **"3D Gaussian Splatting with Deferred Reflection" (3DGS-DR)** tại SIGGRAPH 2024. Dưới đây là phân tích đầy đủ về những gì paper này có thể đóng góp cho **Spec-FastGS**.

---

## 📄 Phân tích Paper: 3DGS-DR (2404.18454v2)

**Tác giả:** Keyang Ye, Qiming Hou, Kun Zhou — SIGGRAPH 2024

---

### 🔍 Ý tưởng cốt lõi của 3DGS-DR

Paper đề xuất thay thế phương pháp tô màu phản chiếu **"per-Gaussian" (forward shading)** bằng kiến trúc **"deferred shading" (tô màu trì hoãn)**:

| Khía cạnh | Forward Shading (GaussianShader) | Deferred Shading (3DGS-DR) |
|---|---|---|
| Tính reflection tại đâu? | Tại từng Gaussian riêng lẻ | Tại từng pixel sau khi splatting |
| Số lượng mẫu phản chiếu | Ít (= số Gaussian) | Nhiều hơn (= số pixel) |
| Chất lượng | Bị nhiễu, có "blob" artifacts | Mịn hơn, ít artifacts |
| Normal map | Per-Gaussian → noisy | Per-pixel → smooth |

**Pipeline 2 giai đoạn:**
1. **Gaussian Splatting Pass**: Splatting bình thường + bake thêm `normal map N(v)` và `reflection strength R(v)` sang screen-space  
2. **Deferred Reflection Pass**: Tính màu cuối bằng: `C'(v) = (1 - R(v))·C(v) + R(v)·E(reflect_dir)`  
   trong đó `E` là **learned environment map**

---

### 💡 3 Đóng góp có thể áp dụng cho Spec-FastGS

---

#### ✅ Đóng góp 1: **Deferred Shading thay vì Forward Shading**

**Vấn đề hiện tại của Spec-FastGS:** Nếu Spec-FastGS đang tính specular color trực tiếp tại từng Gaussian (giống GaussianShader), nó sẽ bị:
- **Oval-shaped blobs** — ranh giới giữa các Gaussian lộ ra
- **Blurry/incorrect reflection** — vì mỗi Gaussian chỉ đóng góp 1 mẫu phản chiếu

**Đề xuất từ 3DGS-DR:** Splatting thêm `normal` và `reflection strength` ra screen-space, sau đó tính reflection per-pixel. Điều này:
- Tăng số mẫu reflection từ O(Gaussians) → O(pixels)
- **Loại bỏ interpolation artifacts** tại ranh giới Gaussian
- Cải thiện gradient propagation cho normal training

> **Tác động dự kiến:** Cải thiện cả visual quality (ít artifacts) và PSNR/SSIM (Table 1 cho thấy 3DGS-DR đạt **+2 đến +3 dB PSNR** so với GaussianShader trên Shiny Blender dataset)

---

#### ✅ Đóng góp 2: **Normal Propagation Strategy**

**Vấn đề:** Normal vector của Gaussian khó optimize chỉ từ color loss vì gradient cực kỳ sparse — chỉ những Gaussian nào đã có normal gần đúng mới nhận được gradient có nghĩa từ environment map bilinear filter.

**Giải pháp của 3DGS-DR:**
1. Khởi tạo với giai đoạn **view-independent bootstrap** (SH order 0, r=0)  
2. Các Gaussian nào đạt `r > 0.1` (đã có normal tốt) → **scale up** diện tích để overlap với Gaussian lân cận  
3. Qua overlap → lan truyền gradient normal tốt sang Gaussian chưa có normal đúng  
4. Kết hợp với **"color sabotage"**: thêm ±10% noise cho Gaussian non-reflective để buộc chúng phải tìm cách fit bằng reflection

**Ablation Study (Table 5):**
- Không có normal propagation: PSNR giảm từ 33.66 → **27.85** (tức **-5.8 dB**)
- Không có color sabotage: PSNR giảm → **30.00** (-3.7 dB)

> **Tác động dự kiến cho Spec-FastGS:** Nếu project đang gặp vấn đề normal không hội tụ trên các bề mặt phản chiếu phức tạp, strategy này có thể cải thiện đáng kể **geometric quality và accuracy của normal map**.

---

#### ✅ Đóng góp 3: **Learned Environment Map cho Light Decoupling**

3DGS-DR dùng một **learnable environment map** E được optimize riêng biệt (detached từ Gaussian), được query bằng reflection direction tính từ normal map.

**Lợi ích:**
- Đơn giản hóa **inverse rendering problem** → ít ill-posed hơn
- Đạt LPIPS tốt nhất (0.511) cho light reconstruction, so với GaussianShader (0.621), ENVIDR (0.615), NVDiffRec (0.636) trong Table 2
- Environment map reconstruct được có **full directional coverage** và ít nhiễu hơn hẳn GaussianShader (xem Fig. 12)

> **Tác động dự kiến:** Nếu Spec-FastGS dùng SH hoặc learned color để handle specular, việc tách riêng environment map có thể giúp **disentangle diffuse vs specular** tốt hơn, cải thiện quantitative metrics trên các scene có ánh sáng phức tạp.

---

### 📊 Kết quả định lượng so sánh (Shiny Blender - trung bình)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time |
|---|---|---|---|---|
| 3DGS (vanilla) | ~27.0 | ~0.944 | ~0.084 | 6 min |
| GaussianShader | ~29.5 | ~0.956 | ~0.067 | 60 min |
| **3DGS-DR (Ours)** | **~32.3** | **~0.971** | **~0.055** | **16 min** |
| Ref-NeRF | ~31.7 | 0.963 | 0.083 | 19h |

3DGS-DR **vượt GaussianShader** về tất cả 3 metrics trong khi **training time chỉ bằng 1/4** (16 phút vs 60 phút).

---

### 🎯 Kết luận: Mức độ phù hợp với Spec-FastGS

| Ý tưởng | Mức độ phù hợp | Độ phức tạp tích hợp | Tác động tiềm năng |
|---|---|---|---|
| Deferred Shading | ⭐⭐⭐⭐⭐ | Trung bình (cần refactor pipeline) | Cao — giảm artifacts rõ rệt |
| Normal Propagation | ⭐⭐⭐⭐ | Thấp (chỉ cần thêm training logic) | Rất cao — +5 dB PSNR |
| Learned Environment Map | ⭐⭐⭐ | Thấp-Trung bình | Trung bình — cải thiện light disentangle |

**Khuyến nghị ưu tiên:** Nếu bottleneck của Spec-FastGS nằm ở **specular surface quality**, thì **Normal Propagation** là ý tưởng dễ tích hợp nhất với ROI cao nhất. Nếu mục tiêu là loại bỏ **Gaussian boundary artifacts**, thì **Deferred Shading** là hướng đi quan trọng hơn — dù đòi hỏi thay đổi pipeline đáng kể.

---

Bạn muốn tôi so sánh sâu hơn 3DGS-DR với một paper cụ thể khác trong thư mục (như **ReflectiveGS** hay **RGS-DR**) để tìm ra chiến lược tổng hợp tốt nhất cho Spec-FastGS không?
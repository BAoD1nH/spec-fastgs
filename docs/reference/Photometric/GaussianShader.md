Listed directory Photometric
Listed directory spec-fastgs

Tôi sẽ đọc file PDF GaussianShader và đồng thời khảo sát codebase của spec-fastgs để có thể phân tích sâu hơn.

Ran command: `cd /home/baodinh/baodinh_thesis/spec-fastgs && python3 -c "
import subprocess
result = subprocess.run(['pdftotext', 'docs/reference/Photometric/2311.17977v1_GaussianShader.pdf', '-'], capture_output=True, text=True)
print(result.stdout[:15000])
"`
Listed directory scene
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2311.17977v1_GaussianShader.pdf - | sed -n '1,300p'`
Viewed gaussian_model.py:1-572
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2311.17977v1_GaussianShader.pdf - | tail -n +200 | head -n 200`
Viewed specular_model.py:1-137
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2311.17977v1_GaussianShader.pdf - | tail -n +400 | head -n 200`
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2311.17977v1_GaussianShader.pdf - | tail -n +600 | head -n 150`
Listed directory utils
Ran command: `pdftotext /home/baodinh/baodinh_thesis/spec-fastgs/docs/reference/Photometric/2311.17977v1_GaussianShader.pdf - | tail -n +750 | head -n 150`
Viewed spec_utils.py:1-189
Viewed loss_utils.py:1-97
Viewed __init__.py:1-149
Viewed train.py:1-440

Tôi đã có đủ thông tin để phân tích sâu. Để tạo một báo cáo đầy đủ, hãy để tôi viết artifact ngay bây giờ.

Created gaussianshader_analysis.md

Đây là bản phân tích đầy đủ. Để tóm tắt các điểm quan trọng nhất:

---

## Kết quả phân tích: GaussianShader → Spec-FastGS

### GaussianShader đề xuất gì?

Thay vì dùng SH thuần túy để model màu sắc view-dependent, GaussianShader dùng **shading function vật lý**:
```
c(ωo) = γ( cd  +  s⊙Ls(ωo,n,ρ)  +  cr(ωo) )
```
với explicit: *diffuse cd*, *specular tint s*, *roughness ρ*, *GGX env-light Ls*, và *SH residual cr*.

---

### Spec-FastGS hiện có gì và thiếu gì?

Spec-FastGS dùng: `Color = SH_color + ASG_MLP_color` — **hoàn toàn không có disentanglement vật lý**.

Tôi phát hiện ra **3 điểm yếu quan trọng** từ source code:

| Gap | Vị trí trong code | Impact |
|---|---|---|
| **Không có normal supervision** | `train.py` – normal chỉ là input, không có loss | ⭐⭐⭐ Cao |
| **`spec_reg = 0.0` bị disabled** | [`train.py` L190](file:///home/baodinh/baodinh_thesis/spec-fastgs/train.py#L190) | ⭐⭐ Trung bình |
| **Không có roughness explicit** | [`spec_utils.py`](file:///home/baodinh/baodinh_thesis/spec-fastgs/utils/spec_utils.py) – ASG học ngầm định | ⭐⭐ Trung bình |

---

### Khuyến nghị ưu tiên (impact/effort ratio)

1. **🥇 Thêm `Lnormal` (Normal-Geometry Consistency Loss)** — ~15 lines, không đổi architecture, dự kiến cải thiện **PSNR +0.5~1.0 dB** trên Shiny Blender / Glossy dataset
2. **🥈 Bật lại `spec_reg`** — 1 dòng code (`* 0.0` → remove), giúp ổn định specular MLP  
3. **🥉 Thêm explicit roughness ρ per-Gaussian** — thay đổi `GaussianModel` + `SpecularNetwork`, medium complexity

GaussianShader đã chứng minh: chỉ cần thêm normal supervision + physical shading, **PSNR trên reflective dataset tăng +1.57 dB** so với 3DGS gốc với overhead training chỉ ~12 phút.
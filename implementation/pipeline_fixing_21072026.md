# Phân tích và đề xuất sửa pipeline Spec-FastGS (21/07/2026)

## 1. Phạm vi

Tài liệu này ghi lại kết quả đối chiếu implementation hiện tại của Spec-FastGS cho ba vấn đề:

1. Reflection Score có thêm Gaussian ngay khi initialization hay chỉ hướng dẫn ADC.
2. VCD/VCP hiện chấm điểm bằng SH hay bằng full appearance SH+ASG, hệ quả và hướng sửa.
3. Reflection Score Extraction có dừng ở confidence map 2D hay đã ánh xạ, tích lũy lên 3D và truyền prior vào point cloud/training.

Tài liệu phân biệt rõ giữa ý tưởng thuật toán, code đã tồn tại, và code thực sự được nối vào training pipeline.

---

## 2. Initialization và vai trò thực tế của Reflection Score

### 2.1 Kết luận

Trong training pipeline hiện tại, Reflection Score **không tự thêm một lượng Gaussian cố định khi initialization**. Số Gaussian ban đầu bằng số điểm của point cloud do dataset loader trả về:

\[
N_{\mathrm{Gaussian,init}}=N_{\mathrm{input\ point\ cloud}}.
\]

- Dữ liệu COLMAP: loader đọc `sparse/0/points3D.ply`.
- Dữ liệu Blender/synthetic: loader đọc `points3d.ply`; nếu chưa có thì tạo ngẫu nhiên 100.000 điểm.

`Scene` sau đó gọi `gaussians.create_from_pcd(scene_info.point_cloud, ...)`. Không có bước ghép thêm Gaussian từ `reflection_prior/` trong đường khởi tạo này.

### 2.2 Reflection Score tác động vào ADC

FastGS tạo pixel metric map từ reconstruction error:

\[
M_{\mathrm{error}}=[E_{\mathrm{L1,norm}}>\tau_{\mathrm{loss}}].
\]

Khi bật `--use_ref_score`, Spec-FastGS bổ sung:

\[
M_{\mathrm{ref}}=[RefScore>\tau_{\mathrm{ref}}],
\]

và dùng hợp OR:

\[
M_{\mathrm{ADC}}=M_{\mathrm{error}}\lor M_{\mathrm{ref}}.
\]

Rasterizer ánh xạ metric map 2D về các Gaussian đã đóng góp vào pixel. Sau đó FastGS ADC mới quyết định clone, split và prune. Reflection Score không trực tiếp sinh Gaussian.

`max_refscore_gaussians` cũng không phải số Gaussian được thêm. Đó là giới hạn tổng số Gaussian mà dưới mức đó RefScore còn được phép hướng dẫn ADC. Với auto budget:

\[
B=\operatorname{clip}(10N_{\mathrm{init}},200000,1000000).
\]

Khi số Gaussian tiến gần budget, ngưỡng RefScore tăng từ `refscore_threshold_min` về phía `refscore_threshold_max`, làm prior ngày càng bảo thủ.

---

## 3. VCD/VCP hiện tại: SH-only scoring và ASG chỉ ảnh hưởng gián tiếp

### 3.1 Đường render dùng để scoring

VCD và VCP đều dùng `compute_gaussian_score_fastgs()`. Hàm này gọi:

```python
render_fastgs(camera, gaussians, pipe, bg, args.mult)
```

mà không truyền `mlp_color`. Trong renderer, khi `mlp_color is None`:

```python
colors_precomp = sh_color
```

Do đó:

\[
C_{\mathrm{VCD/VCP\ scoring}}=C_{\mathrm{SH}},
\]

không phải:

\[
C_{\mathrm{full}}=C_{\mathrm{SH}}+C_{\mathrm{ASG}}.
\]

`_features_asg` không được đọc trực tiếp trong multi-view scoring và Specular MLP không được gọi cho các scoring camera.

### 3.2 Ảnh hưởng gián tiếp của ASG lên VCD

VCD không chỉ dùng multi-view `importance_score`; Gaussian còn phải vượt qua position-gradient qualifier. Gradient này được tích lũy sau backward của training render chính, mà training render chính sau `specular_start_iter` là SH+ASG.

Vì vậy:

\[
\mathrm{VCD\ selection}
=
\underbrace{\mathrm{multi\mbox{-}view\ mask\ from\ SH}}_{\text{không có ASG trực tiếp}}
\land
\underbrace{\mathrm{position\ gradient\ from\ SH+ASG\ training}}_{\text{ASG ảnh hưởng gián tiếp}}.
\]

Khi clone/split, Gaussian con kế thừa `_features_asg` của Gaussian cha. ASG feature đi cùng Gaussian mới nhưng giá trị ASG không trực tiếp quyết định multi-view score.

### 3.3 Ảnh hưởng gián tiếp của ASG lên VCP

`pruning_score` được tính từ photometric loss của SH-only render. Tuy nhiên VCP còn prune theo opacity và kích thước. Opacity, position và scale là các tham số được tối ưu trong training loss SH+ASG, nên ASG vẫn có thể ảnh hưởng gián tiếp đến kết quả prune.

---

## 4. Vấn đề của SH-only VCD/VCP

Tại pixel specular, có thể xấp xỉ:

\[
I_{\mathrm{GT}}=I_{\mathrm{diffuse}}+I_{\mathrm{specular}}.
\]

Nếu ASG đã học tốt:

\[
I_{\mathrm{SH}}+I_{\mathrm{ASG}}\approx I_{\mathrm{GT}},
\]

nhưng SH-only residual vẫn lớn:

\[
|I_{\mathrm{SH}}-I_{\mathrm{GT}}|\approx I_{\mathrm{specular}}.
\]

Hệ thống có thể diễn giải sai lỗi appearance thành thiếu geometry:

```text
SH không tái tạo được highlight
          ↓
SH-only residual cao
          ↓
VCD cho rằng geometry chưa đủ
          ↓
Clone/split thêm Gaussian dù full model đã render tốt
```

RefScore hiện được OR vào cùng metric map nên nguy cơ này có thể bị khuếch đại: cả SH residual và RefScore cùng nhấn mạnh vùng highlight.

Với VCP, false positive nguy hiểm hơn: một Gaussian mang ASG feature hữu ích có thể bị đánh giá xấu chỉ vì SH component của nó không giải thích được highlight.

Mặt khác, SH-only vẫn có ưu điểm ở đầu training: score ít phụ thuộc vào Specular MLP chưa hội tụ và ASG không thể che lấp lỗi geometry bằng view-dependent color. Vì vậy không nên chuyển mù quáng sang full render ngay từ iteration đầu tiên.

---

## 5. Giải pháp đề xuất cho VCD/VCP

### 5.1 VCD theo giai đoạn

Trước khi ASG ổn định, giữ SH-only scoring. Sau ASG warm-up, chuyển dần sang full-render residual:

\[
E_{\mathrm{VCD}}(t)
=(1-\alpha_t)E_{\mathrm{SH}}+\alpha_t E_{\mathrm{full}},
\]

trong đó `alpha_t` tăng từ 0 lên 1 sau `specular_start_iter` và khoảng warm-up.

Giải pháp rõ vai trò hơn là dual-score:

\[
M_{\mathrm{full}}=[E_{\mathrm{full}}>\tau_f],
\]

\[
M_{\mathrm{capacity}}=[E_{\mathrm{SH}}>\tau_s]\lor[RefScore>\tau_r],
\]

\[
M_{\mathrm{VCD}}=M_{\mathrm{full}}\land M_{\mathrm{capacity}}.
\]

`E_full` trả lời câu hỏi “toàn bộ mô hình có còn render sai không?”. `E_SH` và RefScore chỉ đóng vai trò phân loại/ưu tiên capacity, không được tự mình ép sinh thêm geometry khi full model đã render tốt.

### 5.2 VCP nên dùng full render

VCP quyết định xóa tham số của full model, do đó nên đánh giá reconstruction bằng:

\[
E_{\mathrm{VCP}}=|I_{\mathrm{GT}}-(I_{\mathrm{SH}}+I_{\mathrm{ASG}})|.
\]

Final VCP hiện chạy muộn, khi ASG đã có thời gian warm-up, nên lý do duy trì SH-only yếu hơn. Quy tắc an toàn có thể là:

\[
M_{\mathrm{prune}}
=M_{\mathrm{low\ opacity}}
\lor(M_{\mathrm{bad\ full\ score}}\land M_{\mathrm{bad\ visibility}}).
\]

Có thể bảo vệ tạm thời Gaussian có ASG energy/gradient đáng kể trong giai đoạn Specular MLP chưa ổn định.

### 5.3 Cách đưa ASG vào scoring

Không truyền `_features_asg` trực tiếp vào rasterizer vì đó là latent feature, không phải RGB. Với mỗi scoring camera phải decode:

```python
mlp_color = specular_mlp.step(asg_features, viewdir, normal)
render_fastgs(..., mlp_color=mlp_color)
```

ASG color phụ thuộc view direction nên không thể tính một lần rồi dùng chung cho nhiều camera.

Có thể giảm chi phí bằng cách chỉ dùng full render ở pass tạo residual/metric map. Pass rasterization thứ hai chỉ tích lũy `metric_map` về Gaussian có thể không cần chạy lại Specular MLP, sau khi xác nhận CUDA metric accumulation không phụ thuộc màu.

### 5.4 Ablation cần thiết

| Cấu hình | VCD scoring | VCP scoring |
|---|---|---|
| A — baseline hiện tại | SH | SH |
| B | Full | SH |
| C | SH | Full |
| D — đề xuất | Hybrid/dual-score | Full |

Cần theo dõi PSNR/SSIM/LPIPS, số Gaussian cuối, clone/split trong vùng RefScore cao, Gaussian bị final-prune, chất lượng `only_asg`, FPS, thời gian scoring, và cả SH-only/full residual ở vùng specular.

---

## 6. Reflection Score Extraction: trạng thái 2D

### 6.1 Có tạo confidence map riêng cho mỗi ảnh không?

**Có.** `extract_reflection_prior.py` duyệt qua từng training camera, lấy RGB `[H,W,3]` trong `[0,1]`, chạy một trong ba heuristic `tan`, `shafer` hoặc `hybrid`, rồi lưu hai map xám 8-bit:

```text
reflection_prior/<image_name>_ref_score.png
reflection_prior/<image_name>_ref_conf.png
```

- `ref_score`: prior rộng, được dùng chủ yếu cho ADC/geometric coverage.
- `ref_conf`: bản hậu xử lý bảo thủ hơn cho loss weighting và SH/ASG masking.

Hậu xử lý `ref_conf` có thể gồm box smoothing, quantile cutoff và gamma. Với extraction defaults `gamma=1`, `quantile=0`, `smooth_radius=0`, hai map là như nhau. Đây là confidence/prior heuristic theo từng pixel, không phải ground-truth material probability đã hiệu chuẩn.

### 6.2 2D extractor có tự ánh xạ lên 3D không?

**Không.** `extract_reflection_prior.py` kết thúc sau khi ghi các PNG 2D. Nó không dùng depth, không project lên COLMAP point, không kiểm tra occlusion và không tích lũy multi-view.

---

## 7. Script 3D riêng: projection, space carving và multi-view accumulation

Repo có `generate_prior_pcd.py`, là một bước tùy chọn tách rời với 2D extractor.

### 7.1 Những gì script thực sự làm

1. Chỉ chạy nếu dataset có `transforms_train.json`; nếu là real/COLMAP thì thoát ngay.
2. Tạo lưới đều `200^3 = 8.000.000` điểm trong bounding box `[-1.5,1.5]^3`.
3. Chiếu tất cả điểm vào từng camera.
4. Dùng alpha mask: điểm chiếu vào background (`alpha < 0.5`) bị carve.
5. Với điểm chiếu vào foreground, lấy `ref_score` tại pixel và cộng:

\[
S_{3D}(P_i)=\sum_c RefScore_c(\pi_c(P_i))\,V_{i,c}.
\]

6. Sau khi carve, chọn 50.000 điểm có accumulated score cao nhất và 50.000 điểm random từ tập sống sót.
7. Ghi tổng tối đa 100.000 điểm vào `points3d_prior.ply`.

Vì vậy, câu trả lời ở mức script là: **có ánh xạ 2D RefScore lên các giả thuyết điểm 3D và có cộng qua nhiều camera**.

### 7.2 Các giới hạn kỹ thuật

- Đây là visual-hull carving bằng silhouette/alpha, không phải visibility có depth hay z-buffer. Nó không xác nhận điểm nào là bề mặt gần nhất theo tia nhìn.
- Score là tổng, không chia cho `seen_count`, nên điểm được quan sát trong nhiều camera có thể được ưu tiên chỉ vì có nhiều observation.
- Không có kiểm tra multi-view correspondence trên bề mặt thật; tất cả voxel cùng nằm trên một tia foreground có thể nhận cùng score trước carving.
- Tên file prior được hard-code theo `train_<stem>_ref_score.png`, nên phải khớp chính xác quy ước tên từ 2D extractor.
- Biến `sampled_ref` được tính nhưng không dùng; accumulation sau đó sample lại `ref_score`. Đây là dư thừa chứ không làm thay đổi kết quả.

---

## 8. Point cloud có mang reflection prior hay không?

Cần phân biệt ba mức.

### 8.1 Về phân bố không gian: có, nhưng gián tiếp

Nửa point budget là các voxel có accumulated RefScore cao nhất. Do đó `points3d_prior.ply` có mật độ điểm ưu tiên vùng được heuristic cho là specular. Theo nghĩa spatial sampling distribution, point cloud mang prior gián tiếp.

### 8.2 Về thuộc tính từng điểm: không

PLY đầu ra chỉ ghi:

```text
x, y, z, nx, ny, nz, red, green, blue
```

`accumulated_ref_score`, nhãn `is_specular`, confidence, hay ASG allocation không được ghi. Màu cũng được khởi tạo ngẫu nhiên. Sau khi file được ghi, không thể biết điểm nào thuộc top-RefScore chỉ từ schema của PLY.

Vì vậy không thể nói mỗi Gaussian được initialization với một reflection confidence. Prior chỉ được mã hóa gián tiếp qua việc điểm nào được chọn vào point cloud.

### 8.3 Về training pipeline hiện tại: chưa được sử dụng tự động

`generate_prior_pcd.py` ghi `points3d_prior.ply`, nhưng:

- Blender loader chỉ đọc `points3d.ply`.
- COLMAP loader chỉ đọc `sparse/0/points3D.ply`.
- Không có run script nào tự gọi `generate_prior_pcd.py` hay chuyển output của nó thành input PLY cho loader.

Do đó, pipeline mặc định hiện nay là:

```text
RGB → 2D RefScore/RefConf → nạp theo camera → hướng dẫn ADC/loss/masking
```

chứ chưa phải:

```text
RGB → 2D prior → multi-view 3D prior → prior-aware point cloud → initialization
```

Muốn dùng `points3d_prior.ply` hiện tại phải chủ động đổi tên/copy thành `points3d.ply` cho synthetic, hoặc sửa loader. Cách này vẫn chỉ truyền spatial sampling prior, không truyền per-point score.

---

## 9. Đề xuất hoàn thiện prior 2D → 3D

Nếu mục tiêu là prior-aware initialization thực sự, cần:

1. Nối `generate_prior_pcd.py` vào preprocessing/run script và loader bằng flag rõ ràng, không ghi đè point cloud gốc ngầm định.
2. Lưu `ref_score_3d` và/hoặc `is_specular_prior` trong PLY/sidecar file, thay vì chỉ dùng score để top-k rồi vứt bỏ.
3. Truyền per-point prior vào `GaussianModel.create_from_pcd()` để initialization ASG/gating/budget có thể dùng nó.
4. Chuẩn hóa theo valid observation count, hoặc lưu cả `score_sum`, `seen_count`, `score_mean` và `score_max` để tránh visibility-count bias.
5. Dùng depth/z-buffer hoặc project COLMAP/surface points thay cho chỉ silhouette carving, để score được gán cho bề mặt nhìn thấy thay vì toàn bộ voxel trên tia foreground.
6. Hỗ trợ real/COLMAP thay vì thoát khi không có `transforms_train.json`.
7. Quy định rõ prior 3D dùng cho density allocation, ASG activation, hay cả hai; không đồng nhất “điểm được sample nhiều” với “Gaussian đã được gán material specular”.

---

## 10. Kết luận cuối

1. Reflection Score Extraction hiện chắc chắn tạo `ref_score` và `ref_conf` 2D riêng cho mỗi training image.
2. 2D extractor không tự làm projection hay multi-view accumulation.
3. Một script 3D riêng có thực hiện projection, alpha carving và cộng RefScore qua camera cho synthetic data.
4. Point cloud do script này sinh ra mang reflection prior **gián tiếp qua phân bố/mật độ điểm**, nhưng **không mang per-point reflection score hoặc specular label**.
5. `points3d_prior.ply` **chưa được training loader sử dụng tự động**, nên pipeline mặc định vẫn khởi tạo từ point cloud gốc và chỉ dùng 2D prior để hướng dẫn ADC/các cơ chế training tùy chọn.
6. VCD/VCP multi-view scoring hiện là SH-only. Hướng sửa khuyến nghị là VCD hybrid/dual-score sau ASG warm-up và VCP full SH+ASG, thay vì dùng SH-only cho cả hai trong suốt training.

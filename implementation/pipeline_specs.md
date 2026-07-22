Có. Trong phiên bản hiện tại của `spec-fastgs`, “reflection score extraction” là một bước tiền xử lý offline: mỗi ảnh huấn luyện được biến thành một bản đồ xám 2D, trong đó giá trị gần `1` biểu thị pixel có nhiều khả năng chứa phản xạ gương, còn gần `0` biểu thị vùng thiên về diffuse.

Điểm quan trọng là: đây là một optical prior dựa trên heuristic màu sắc, không phải ground truth vật liệu và cũng không phải kết quả của một mạng học sâu.

## 1. Luồng xử lý tổng quát

```text
Ảnh RGB của từng camera
        ↓
Chuẩn hóa RGB về [0,1]
        ↓
Tính tín hiệu sáng / độ bão hòa
        ↓
Tan-Ikeuchi, Shafer-Klinker hoặc Hybrid
        ↓
Chuẩn hóa riêng từng ảnh về [0,1]
        ↓
Tạo ref_score và ref_conf
        ↓
Lưu thành PNG 8-bit trong reflection_prior/
        ↓
Nạp lại khi training để hướng dẫn densification/loss/masking
```

Mã chính nằm trong [extract_reflection_prior.py](/home/baodinh/baodinh_thesis/spec-fastgs/extract_reflection_prior.py:16).

## 2. Đầu vào

Script khởi tạo `Scene` để sử dụng chính loader camera của Spec-FastGS:

```python
gaussians = GaussianModel(0)
scene = Scene(dataset, gaussians)
train_cameras = scene.getTrainCameras()
```

Với mỗi camera, nó lấy:

```python
cam.original_image  # [3,H,W], giá trị thuộc [0,1]
```

rồi chuyển thành:

```python
img01  # [H,W,3]
```

Nhờ vậy, prior được tạo đúng ở độ phân giải mà dataset loader đang sử dụng, chẳng hạn `images_8`.

## 3. Ba phương pháp tính Reflection Score

### 3.1 Tan–Ikeuchi — mặc định hiện tại

Đây là phương pháp mặc định trong script chạy:

```bash
REF_PRIOR_METHOD=tan
```

Với mỗi pixel RGB, hệ thống tính:

\[
I_{\min}=\min(R,G,B), \qquad
I_{\max}=\max(R,G,B)
\]

Pixel chỉ được xem là ứng viên phản xạ nếu:

\[
I_{\min}>0.35
\quad\land\quad
I_{\max}>0.6
\]

Điểm thô là:

\[
S_{\text{Tan}} =
\begin{cases}
I_{\min}, & I_{\min}>0.35 \land I_{\max}>0.6\\
0, & \text{ngược lại}
\end{cases}
\]

Sau đó chia cho giá trị lớn nhất trong ảnh:

\[
RefScore(x,y)=
\frac{S_{\text{Tan}}(x,y)}
{\max_{u,v}S_{\text{Tan}}(u,v)}
\]

Ý tưởng của việc dùng `Imin` là: một highlight trắng hoặc gần trắng phải sáng đồng thời ở cả ba kênh. Một pixel đỏ rất sáng có thể có `Imax` cao nhưng `Imin` thấp, nên sẽ không dễ bị xem là specular.

Phần này được cài đặt tại [extract_reflection_prior.py](/home/baodinh/baodinh_thesis/spec-fastgs/extract_reflection_prior.py:16).

### 3.2 Shafer–Klinker

Phương pháp này dựa trên xấp xỉ Dichromatic Reflection Model: highlight thường có cường độ cao và độ bão hòa thấp.

Độ bão hòa được xấp xỉ bằng:

\[
Sat = 1-\frac{I_{\min}}{I_{\max}+\epsilon}
\]

Ứng viên specular phải thỏa:

\[
I_{\max}>0.7
\quad\land\quad
Sat<0.2
\]

Điểm:

\[
S_{\text{Shafer}}
=I_{\max}(1-Sat)
\]

Có thể thấy:

\[
I_{\max}(1-Sat)
\approx I_{\min}
\]

nhưng Shafer bổ sung điều kiện rõ ràng về độ bão hòa. Vì vậy, nó bảo thủ hơn với những vùng sáng có màu mạnh.

### 3.3 Hybrid

Hybrid tạo prior mềm hơn bằng cách phối hợp bốn tín hiệu:

\[
S =
0.35S_{\text{Tan-soft}}
+0.35S_{\text{Shafer-soft}}
+0.20S_{\text{gray-bright}}
+0.10S_{\text{local}}
\]

Trong đó:

- `Tan-soft`: phiên bản sigmoid mềm của các ngưỡng Tan–Ikeuchi.
- `Shafer-soft`: pixel sáng và ít bão hòa.
- `gray-bright`: mức sáng trung tính,

\[
S_{\text{gray}}=I_{\max}(1-Sat)
\]

- `local-highlight`: độ sáng vượt quá trung bình cục bộ,

\[
S_{\text{local}} =
\max(I_{\max}-\operatorname{BoxBlur}(I_{\max}),0)
\]

Thành phần local contrast giúp phân biệt highlight nhỏ, cục bộ với một bề mặt diffuse trắng và sáng đồng đều.

Sigmoid mềm được sử dụng là:

\[
\operatorname{softstep}(x;c,t)
=
\frac{1}{1+\exp(-(x-c)/t)}
\]

với `temperature = 0.06`.

## 4. `ref_score` và `ref_conf` khác nhau thế nào?

Mỗi ảnh tạo ra hai file:

```text
<image_name>_ref_score.png
<image_name>_ref_conf.png
```

### `ref_score`

Đây là prior rộng, được chuẩn hóa về `[0,1]`. Nó chủ yếu thích hợp để hướng dẫn geometric coverage và densification.

### `ref_conf`

Đây là phiên bản bảo thủ hơn dành cho những thao tác mà false positive gây hại lớn hơn, ví dụ:

- tăng trọng số loss trong vùng specular;
- xác định vùng giảm gradient SH;
- phân tách vai trò SH và ASG.

Quá trình hậu xử lý gồm:

1. Chuẩn hóa.
2. Box blur tùy chọn.
3. Loại phần dưới một quantile tùy chọn.
4. Áp dụng gamma.
5. Chuẩn hóa lại.

Nếu dùng cấu hình mặc định của script:

```bash
REF_CONF_GAMMA=1.0
REF_CONF_QUANTILE=0.0
REF_CONF_SMOOTH_RADIUS=0
```

thì:

\[
ref\_conf = ref\_score
\]

Tức là hai ảnh đầu ra giống nhau. Chúng chỉ khác khi bật hậu xử lý.

## 5. Cách quantile và gamma làm confidence bảo thủ hơn

Giả sử `quantile = 0.85`. Hệ thống lấy phân vị thứ 85:

\[
p=Q_{0.85}(S)
\]

rồi ánh xạ:

\[
S'=
\operatorname{clip}
\left(
\frac{S-p}{S_{\max}-p+\epsilon},
0,1
\right)
\]

Như vậy phần lớn pixel có score thấp bị đưa về `0`; chỉ vùng score cao nhất còn lại.

Sau đó:

\[
S''=(S')^\gamma
\]

Với \(\gamma>1\), các giá trị trung bình tiếp tục bị giảm. Ví dụ:

\[
0.5^{1.5}\approx0.354
\]

Trong training còn có một lớp hậu xử lý khác với mặc định `refscore_conf_quantile=0.85` và `refscore_conf_gamma=1.5`, nhưng nó chủ yếu đóng vai trò fallback nếu file `_ref_conf.png` không tồn tại.

## 6. Đầu ra được lưu như thế nào?

Score `[0,1]` được lượng tử hóa thành PNG 8-bit:

\[
PNG(x,y)=\lfloor 255\cdot RefScore(x,y)\rfloor
\]

Thư mục kết quả:

```text
<dataset>/
└── reflection_prior/
    ├── image_001_ref_score.png
    ├── image_001_ref_conf.png
    ├── image_002_ref_score.png
    └── image_002_ref_conf.png
```

Tên file phải khớp chính xác `cam.image_name`, vì training tìm prior bằng tên đó.

Script điều khiển nằm tại [run_extract_reflection_prior.sh](/home/baodinh/baodinh_thesis/spec-fastgs/run_extract_reflection_prior.sh:1). Script cũng backup thư mục prior cũ trước khi ghi lại.

## 7. Reflection Score được dùng trong training ra sao?

Khi bật `--use_ref_score`, [train.py](/home/baodinh/baodinh_thesis/spec-fastgs/train.py:184) nạp hai map và gắn chúng vào từng camera:

```python
cam.ref_score
cam.ref_score_conf
```

Nếu kích thước không khớp ảnh training, prior được resize bằng bilinear interpolation.

### Hướng dẫn densification

FastGS ban đầu đánh dấu pixel cần densify bằng photometric error:

\[
M_{\text{error}}=
[\operatorname{L1}_{pixel}>\tau_{\text{loss}}]
\]

Spec-FastGS bổ sung mask từ reflection score:

\[
M_{\text{ref}}=[RefScore>\tau_{\text{ref}}]
\]

Hai mask được hợp bằng OR:

\[
M=\max(M_{\text{error}},M_{\text{ref}})
\]

Sau rasterization, mask 2D được quy ngược về các Gaussian đóng góp vào những pixel đó. Vì vậy Gaussian nằm trong vùng có phản xạ dễ được clone/split hơn.

Điểm cần nhấn mạnh: reflection score không trực tiếp sinh Gaussian. Nó chỉ mở rộng `metric_map` mà FastGS ADC dùng để quyết định Gaussian nào đáng densify. Phần này nằm trong [fast_utils.py](/home/baodinh/baodinh_thesis/spec-fastgs/utils/fast_utils.py:82).

Ngưỡng reflection còn tăng dần theo mức sử dụng budget:

\[
r=\operatorname{clip}\left(
\frac{N_{\text{current}}}{N_{\text{budget}}},0,1
\right)
\]

\[
strength=
\max((1-r)^p,\ strength_{\min})
\]

\[
\tau_{\text{ref}}
=
\tau_{\min}
+(1-strength)(\tau_{\max}-\tau_{\min})
\]

Đầu training, ngưỡng gần `0.5`, nên prior tác động rộng. Khi số Gaussian tiến gần budget, ngưỡng tiến về `0.9`, nên chỉ highlight có độ tin cậy cao tiếp tục tác động.

### Các công dụng tùy chọn khác

`ref_conf` còn có thể:

- tăng trọng số L1 ở vùng phản xạ:

\[
w(x,y)=1+\lambda_{\text{spec-L1}}RefConf(x,y)
\]

- chiếu mask 2D về Gaussian để giảm gradient các hệ số SH bậc cao trong vùng specular, nhường phần view-dependent cho ASG;
- cập nhật adaptive prior bằng residual giữa ảnh ground truth và ảnh chỉ render bằng base/SH:

\[
RefScore_{\text{adaptive}}
=
Residual_{\text{normalized}}\cdot RefScore_{\text{static}}
\]

sau đó trộn EMA với prior hiện tại.

Các cơ chế này đều là tùy chọn; cấu hình mặc định trong `arguments` đang tắt chúng hoặc đặt trọng số bằng `0`.

## 8. Có phải extraction đã tạo prior 3D không?

Không. `extract_reflection_prior.py` chỉ tạo các bản đồ 2D độc lập cho từng camera. Nó không:

- dùng depth;
- kiểm tra occlusion;
- đối chiếu cùng một điểm giữa nhiều view;
- suy ra normal hoặc roughness;
- tạo nhãn vật liệu thật.

Repo có bước riêng [generate_prior_pcd.py](/home/baodinh/baodinh_thesis/spec-fastgs/generate_prior_pcd.py:58), có thể chiếu các map 2D lên lưới 3D, space carving theo alpha và cộng score qua nhiều view. Tuy nhiên bước này:

- chỉ chạy cho dữ liệu synthetic có `transforms_train.json`;
- yêu cầu alpha mask;
- bỏ qua dữ liệu COLMAP thực;
- tạo `points3d_prior.ply`;
- không thuộc chính thao tác 2D “reflection score extraction”.

## 9. Hạn chế quan trọng

Reflection score thực chất phát hiện “sáng và gần trung tính”, nên có thể:

- nhận nhầm tường trắng, đèn, bầu trời hoặc vùng overexposure;
- bỏ sót phản xạ có màu;
- bỏ sót kim loại tối hoặc highlight không đủ sáng;
- cho score không so sánh tuyệt đối tốt giữa các ảnh, vì mỗi ảnh được chuẩn hóa theo maximum riêng;
- tạo ngưỡng thiếu ổn định nếu exposure giữa các camera thay đổi.

Vì vậy cách hiểu chính xác nhất là:

> Reflection Score không khẳng định pixel là vật liệu phản xạ; nó đánh dấu nơi mô hình nên dành thêm sự chú ý vì tín hiệu ảnh có đặc điểm giống highlight specular.

Trong cấu hình hiện tại, tác dụng cốt lõi nhất của nó là hướng dẫn FastGS densify có chọn lọc ở các vùng nghi ngờ specular, còn ASG vẫn được học thông qua reconstruction loss.

---

Hãy trình bày cho tôi biết là spec-fastgs có thêm 1 lượng gaussian nhất định vào khi initialization không hay là chỉ tác động vào qui trình ADC thôi?

Trong pipeline training hiện tại, Spec-FastGS không tự thêm một lượng Gaussian mới dựa trên Reflection Score tại initialization. Reflection Score chủ yếu tác động vào quá trình ADC/densification sau khi training bắt đầu.

## 1. Initialization thực tế

Khi bắt đầu training:

```python
gaussians = GaussianModel(...)
scene = Scene(dataset, gaussians)
```

`Scene` lấy point cloud của dataset và gọi:

```python
gaussians.create_from_pcd(scene_info.point_cloud, ...)
```

tại [scene/__init__.py](/home/baodinh/baodinh_thesis/spec-fastgs/scene/__init__.py:135).

Số Gaussian ban đầu bằng đúng số điểm trong point cloud được loader đọc:

\[
N_{\text{Gaussian, init}}=N_{\text{point cloud}}
\]

### Đối với dataset COLMAP

Loader luôn đọc:

```text
<dataset>/sparse/0/points3D.ply
```

xem [dataset_readers.py](/home/baodinh/baodinh_thesis/spec-fastgs/scene/dataset_readers.py:208).

Do đó:

\[
N_{\text{init}}=N_{\text{COLMAP points}}
\]

Reflection Score không bổ sung Gaussian vào tập này.

### Đối với Blender/synthetic

Loader đọc:

```text
<dataset>/points3d.ply
```

Nếu file chưa có, nó sinh ngẫu nhiên đúng `100,000` điểm:

```python
num_pts = 100_000
```

xem [dataset_readers.py](/home/baodinh/baodinh_thesis/spec-fastgs/scene/dataset_readers.py:290).

Một lần nữa, việc này không sử dụng Reflection Score.

## 2. Reflection Score tác động ở ADC như thế nào?

Trong FastGS thông thường, pixel được đưa vào metric map nếu reconstruction error cao:

\[
M_{\mathrm{error}}
=
[\operatorname{L1}_{pixel}>\tau_{\mathrm{loss}}]
\]

Khi bật `--use_ref_score`, Spec-FastGS thêm:

\[
M_{\mathrm{ref}}
=
[RefScore>\tau_{\mathrm{ref}}]
\]

Sau đó hợp hai mask:

\[
M_{\mathrm{ADC}}
=
M_{\mathrm{error}}\lor M_{\mathrm{ref}}
\]

Trong code:

```python
metric_map = (l1_loss_norm > args.loss_thresh).int()

ref_mask = (
    my_viewpoint_cam.ref_score.cuda()
    > ref_score_threshold
).int()

metric_map = torch.max(metric_map, ref_mask)
```

xem [fast_utils.py](/home/baodinh/baodinh_thesis/spec-fastgs/utils/fast_utils.py:82).

Rasterizer chiếu mask pixel này ngược về các Gaussian đóng góp vào pixel. Sau đó `densify_and_prune_fastgs()` vẫn là thành phần thực sự quyết định:

- Gaussian nào được clone;
- Gaussian nào được split;
- Gaussian nào bị prune.

Vì vậy quan hệ chính xác là:

```text
Reflection Score
      ↓
Mở rộng metric map của ADC
      ↓
Tăng khả năng Gaussian ở vùng specular được chọn
      ↓
FastGS ADC thực hiện clone/split
```

Reflection Score không tự gọi hàm thêm Gaussian và cũng không trực tiếp tạo Gaussian mới.

## 3. `max_refscore_gaussians` có phải số Gaussian được thêm lúc init không?

Không.

Sau initialization, code đọc số Gaussian hiện có:

```python
initial_gaussians = gaussians.get_xyz.shape[0]
configure_refscore_budget(opt, initial_gaussians)
```

Nếu `max_refscore_gaussians=-1`, budget tự động được tính:

\[
B=
\operatorname{clip}
\left(
10N_{\text{init}},
200\,000,
1\,000\,000
\right)
\]

Nhưng \(B\) không phải số Gaussian được thêm. Nó là giới hạn tổng số Gaussian mà dưới mức đó Reflection Score còn được phép tham gia hướng dẫn ADC.

Ví dụ:

- COLMAP ban đầu: \(N_{\text{init}}=155\,000\);
- \(10N_{\text{init}}=1\,550\,000\);
- sau khi clamp: \(B=1\,000\,000\).

Điều này không có nghĩa Spec-FastGS thêm 845.000 Gaussian. Nó chỉ có nghĩa prior được phép hỗ trợ ADC trong khi:

\[
N_{\text{current}}<1\,000\,000
\]

Số Gaussian thực sự tăng bao nhiêu vẫn do FastGS ADC quyết định qua clone/split và có thể đồng thời giảm do pruning.

## 4. Ngoại lệ: `generate_prior_pcd.py`

Repo có [generate_prior_pcd.py](/home/baodinh/baodinh_thesis/spec-fastgs/generate_prior_pcd.py:58), tạo point cloud `points3d_prior.ply` gồm tối đa:

- 50.000 điểm base;
- 50.000 điểm có accumulated reflection score cao;
- tổng cộng 100.000 điểm.

Thoạt nhìn, đây có vẻ là reflection-guided initialization. Nhưng pipeline training hiện tại không đọc file này:

```text
points3d_prior.ply
```

Synthetic loader chỉ đọc:

```text
points3d.ply
```

COLMAP loader chỉ đọc:

```text
sparse/0/points3D.ply
```

Vì vậy, chỉ chạy `generate_prior_pcd.py` không làm thay đổi initialization. Muốn dùng nó, người chạy phải chủ động đổi tên/copy nó thành `points3d.ply`, hoặc sửa dataset loader. Hơn nữa, script này hiện chỉ hỗ trợ synthetic dataset có `transforms_train.json` và alpha mask.

## Kết luận

Trong code hiện tại:

\[
\boxed{\text{Reflection Score không thêm Gaussian tại initialization}}
\]

Nó hoạt động chủ yếu như tín hiệu hướng dẫn ADC:

\[
\boxed{
\text{RefScore}
\rightarrow
\text{metric map}
\rightarrow
\text{FastGS clone/split/prune}
}
\]

`generate_prior_pcd.py` là một cơ chế initialization thử nghiệm/tùy chọn, nhưng chưa được nối tự động vào training loader. Vì thế mô tả chính xác cho pipeline mặc định là: khởi tạo giống FastGS từ point cloud gốc, sau đó Reflection Score mới định hướng densification trong quá trình ADC.

---
Hãy trình bày cho tôi biết vcd và vcp hiện tại có bị ảnh hưởng bởi asg_features không, hay vcd và vcp chỉ đang dùng mỗi sh

Kết luận chính xác là:

> Phần multi-view scoring của VCD và VCP hiện tại chỉ dùng ảnh render từ SH, không đưa `asg_features` hay màu specular của ASG vào. Tuy nhiên, VCD/VCP vẫn có thể bị ASG ảnh hưởng gián tiếp thông qua gradient, opacity và trạng thái Gaussian được học trong training chính.

## 1. Vì sao nói VCD/VCP scoring chỉ dùng SH?

Cả VCD và VCP đều lấy score từ:

```python
compute_gaussian_score_fastgs(...)
```

Trong hàm này, ảnh đánh giá được render như sau:

```python
render_image = render_fastgs(
    my_viewpoint_cam,
    gaussians,
    pipe,
    bg,
    args.mult
)["render"]
```

Không có đối số:

```python
mlp_color=...
```

Do đó `mlp_color` nhận giá trị mặc định `None`.

Trong renderer:

```python
if mlp_color is not None:
    colors_precomp = sh_color + mlp_color
else:
    colors_precomp = sh_color
```

Vì vậy:

\[
C_{\text{VCD/VCP}}=C_{\text{SH}}
\]

chứ không phải:

\[
C_{\text{VCD/VCP}}=C_{\text{SH}}+C_{\text{ASG}}
\]

Phần này có thể thấy tại:

- [fast_utils.py](/home/baodinh/baodinh_thesis/spec-fastgs/utils/fast_utils.py:73)
- [gaussian_renderer/__init__.py](/home/baodinh/baodinh_thesis/spec-fastgs/gaussian_renderer/__init__.py:121)

## 2. `asg_features` được dùng ở đâu?

Trong training chính, sau `specular_start_iter`, code lấy:

```python
asg_feat = gaussians.get_asg_features
```

và đưa qua Specular MLP:

```python
spec_sparse = specular_mlp.step(
    asg_feat[vis_indices],
    viewdir[vis_indices],
    normal.detach()[vis_indices],
)
```

Sau đó tạo `mlp_color` và render:

```python
render_fastgs(..., mlp_color=mlp_color)
```

Do đó ảnh dùng để tối ưu photometric loss chính là:

\[
C_{\text{train}}=C_{\text{SH}}+C_{\text{ASG}}
\]

Nhưng khi đến bước tính score VCD/VCP, hệ thống render lại riêng bằng:

\[
C_{\text{score}}=C_{\text{SH}}
\]

## 3. VCD hiện tại có bị ASG ảnh hưởng không?

### Phần multi-view VCD: không trực tiếp

VCD xây dựng `metric_map` từ residual của ảnh SH-only:

\[
E_{\text{SH}}(x,y)
=
\left|
I_{\text{SH}}(x,y)-I_{\text{GT}}(x,y)
\right|
\]

Sau đó:

\[
M_{\text{error}}
=
[E_{\text{SH,norm}}>\tau_{\text{loss}}]
\]

Nếu bật Reflection Score:

\[
M_{\text{metric}}
=
M_{\text{error}}\lor M_{\text{ref}}
\]

Rasterizer tích lũy số view mà mỗi Gaussian đóng góp vào các pixel được đánh dấu. Kết quả là `importance_score`, sau đó dùng điều kiện:

```python
metric_mask = importance_score > 5
```

Toàn bộ nhánh này không đọc `_features_asg` và không gọi `specular_mlp`.

### Nhưng VCD bị ASG ảnh hưởng gián tiếp qua gradient

Để được clone hoặc split, một Gaussian không chỉ cần vượt qua multi-view metric. Nó còn phải vượt qua gradient qualifier:

```python
grad_qualifiers =
    norm(xyz_gradient_accum / denom) >= grad_thresh

grad_qualifiers_abs =
    norm(xyz_gradient_accum_abs / denom) >= grad_abs_thresh
```

Các gradient này được tích lũy từ `viewspace_point_tensor.grad` của vòng training chính:

```python
loss.backward()
gaussians.add_densification_stats(...)
```

Mà loss chính được tính trên ảnh:

\[
I_{\text{SH+ASG}}
\]

Do đó ASG có thể làm thay đổi residual và gradient vị trí trong forward/backward chính, rồi gián tiếp làm Gaussian vượt hoặc không vượt điều kiện clone/split.

Có thể mô tả VCD hiện tại như sau:

\[
\text{VCD selection}
=
\underbrace{\text{multi-view mask từ SH-only}}_{\text{không trực tiếp có ASG}}
\land
\underbrace{\text{position gradient từ SH+ASG training}}_{\text{có ảnh hưởng ASG gián tiếp}}
\]

## 4. VCP hiện tại có bị ASG ảnh hưởng không?

### `pruning_score`: chỉ dùng SH

VCP tính:

```python
photometric_loss = compute_photometric_loss(
    camera,
    render_image
)
```

Trong đó `render_image` là SH-only.

Sau đó:

\[
S_i^{raw}
=
\sum_c
L_{\text{photo,SH}}^{(c)}
\cdot count_i^{(c)}
\]

và chuẩn hóa thành `pruning_score`.

Ở final pruning:

```python
scores_mask = pruning_score > 0.9
final_prune = opacity_mask | scores_mask
```

Vì vậy nhánh multi-view reconstruction score của VCP không dùng ASG.

### Nhưng opacity có thể bị ASG ảnh hưởng gián tiếp

VCP còn prune dựa trên:

```python
self.get_opacity < min_opacity
```

Opacity được học trong training chính với ảnh `SH + ASG`. Vì thế ASG có thể làm thay đổi gradient tác động lên opacity và gián tiếp ảnh hưởng Gaussian nào bị loại.

Ngoài ra, scale, position và geometry hiện tại cũng là kết quả của toàn bộ quá trình tối ưu chung.

Do đó:

\[
\text{VCP}
=
\underbrace{\text{pruning score từ SH-only}}_{\text{không có ASG trực tiếp}}
\lor
\underbrace{\text{opacity học từ SH+ASG}}_{\text{ASG ảnh hưởng gián tiếp}}
\]

## 5. Khi clone/split, ASG feature có được giữ lại không?

Có.

Khi một Gaussian được chọn để clone, code sao chép cả:

```python
new_features_asg = self._features_asg[selected_pts_mask]
```

rồi truyền vào `densification_postfix()`.

Vì vậy:

- `asg_features` không quyết định trực tiếp multi-view score;
- nhưng Gaussian con sinh ra sẽ kế thừa `asg_features` của Gaussian cha;
- sau đó các feature này tiếp tục được tối ưu.

Điều này cũng đúng về mặt ý tưởng đối với split: ASG capacity đi cùng Gaussian mới, dù tiêu chí lựa chọn không trực tiếp đọc giá trị ASG.

## 6. Tóm tắt

| Thành phần | Dùng SH | Dùng ASG trực tiếp | ASG ảnh hưởng gián tiếp |
|---|---:|---:|---:|
| Ảnh training chính | Có | Có | — |
| Pixel-error map của VCD | Có | Không | Không |
| Multi-view `importance_score` | Có | Không | Không |
| Gradient qualifier của VCD | Có | Có qua loss chính | Có |
| `pruning_score` của VCP | Có | Không | Không |
| Opacity pruning | Có | Có qua loss chính | Có |
| Sao chép feature khi clone/split | Có | Có, kế thừa ASG | — |

Nói ngắn gọn:

\[
\boxed{
\text{VCD/VCP scoring hiện tại là SH-only}
}
\]

nhưng không nên nói toàn bộ VCD/VCP hoàn toàn độc lập với ASG, bởi gradient densification và opacity pruning được hình thành trong quá trình tối ưu ảnh `SH + ASG`.

Một hệ quả đáng chú ý là: nếu ASG đã giải thích tốt highlight nhưng SH chưa giải thích được, VCD/VCP scoring vẫn nhìn thấy residual SH-only lớn. Đây có vẻ là lựa chọn có chủ đích nhằm đo phần geometry/base representation chưa giải thích được, nhưng nó cũng có nguy cơ tiếp tục densify hoặc đánh giá xấu một vùng mà full model `SH + ASG` thực ra đã render tốt.

---
phân tích tiếp cho tôi biết ở khâu "Reflection Score Extraction" sau khi đưa ảnh đầu vào qua các 2d heuristic model để phát hiện vùng phản xạ thì có tạo ra bản đồ độ tin cậy 2D (Mỗi ảnh) không? Xong rồi có ánh xạ lên điểm 3D và tích lũy đa góc nhìn không? Rồi đám mây điểm có mang tiền nghiệm phản xạ hay không?
Mình đã ghi toàn bộ phân tích và đề xuất vào [pipeline_fixing_21072026.md](/home/baodinh/baodinh_thesis/spec-fastgs/implementation/pipeline_fixing_21072026.md).

Về Reflection Score Extraction, kết luận theo code hiện tại như sau:

## 1. Có tạo confidence map 2D cho từng ảnh không?

Có. Mỗi training image được đưa qua một trong ba heuristic:

- Tan–Ikeuchi;
- Shafer–Klinker;
- Hybrid.

Sau đó tạo hai bản đồ:

```text
reflection_prior/<image_name>_ref_score.png
reflection_prior/<image_name>_ref_conf.png
```

Trong đó:

- `ref_score` là prior rộng, chủ yếu hướng dẫn ADC;
- `ref_conf` là confidence map bảo thủ hơn, có thể dùng cho loss weighting và SH/ASG masking.

Với tham số extraction mặc định hiện tại:

```text
gamma = 1
quantile = 0
smooth_radius = 0
```

thì `ref_conf` thực tế giống `ref_score`.

## 2. Sau đó có tự động ánh xạ lên 3D không?

Không phải trong `extract_reflection_prior.py`.

Pipeline extraction chính dừng tại:

```text
RGB image
    ↓
2D heuristic
    ↓
ref_score.png + ref_conf.png
```

Nó không thực hiện:

- chiếu lên COLMAP point cloud;
- dùng depth;
- kiểm tra occlusion;
- tích lũy score giữa các camera;
- tạo thuộc tính reflection cho Gaussian.

## 3. Repo có code ánh xạ và tích lũy đa góc nhìn không?

Có, nhưng nằm trong script riêng:

```text
generate_prior_pcd.py
```

Script này:

1. Yêu cầu synthetic dataset có `transforms_train.json`.
2. Tạo lưới \(200^3=8.000.000\) điểm 3D.
3. Chiếu các điểm vào từng camera.
4. Dùng alpha mask để space carving.
5. Lấy Reflection Score tại pixel tương ứng.
6. Cộng score qua nhiều camera:

\[
S_{3D}(P_i)
=
\sum_c RefScore_c(\pi_c(P_i))V_{i,c}
\]

7. Chọn:

   - 50.000 điểm có accumulated score cao nhất;
   - 50.000 điểm ngẫu nhiên từ visual hull.

8. Ghi thành:

```text
points3d_prior.ply
```

Vì vậy, ở mức script độc lập, repo có triển khai phép chiếu và tích lũy đa góc nhìn.

## 4. Đây có phải visibility/geometry 3D chính xác không?

Chưa hoàn toàn.

Script đang dùng alpha silhouette carving, không có depth hoặc z-buffer. Vì vậy, nếu nhiều voxel cùng nằm trên một tia foreground, chúng có thể cùng nhận Reflection Score mặc dù chỉ một vị trí thực sự nằm trên bề mặt nhìn thấy.

Ngoài ra, score đang dùng tổng:

\[
S_i=\sum_c S_{i,c}
\]

chứ không chuẩn hóa theo số camera nhìn thấy:

\[
\bar S_i=
\frac{\sum_c S_{i,c}}{\operatorname{seenCount}_i}
\]

Do đó điểm được quan sát bởi nhiều camera có thể được ưu tiên chỉ vì có nhiều observation hơn.

Nói chính xác, đây là:

> Reflection-score-guided visual-hull sampling,

chưa phải phép ánh xạ reflection confidence lên bề mặt 3D có occlusion chính xác.

## 5. Point cloud có mang reflection prior không?

Có theo nghĩa gián tiếp, nhưng không có thuộc tính reflection rõ ràng.

`points3d_prior.ply` mang prior thông qua phân bố điểm:

- vùng accumulated RefScore cao được lấy nhiều điểm hơn;
- do đó vùng nghi ngờ specular có mật độ initialization cao hơn.

Nhưng PLY chỉ lưu:

```text
x, y, z
nx, ny, nz
red, green, blue
```

Nó không lưu:

```text
ref_score_3d
ref_confidence
is_specular
ASG allocation
```

Màu RGB trong PLY còn được khởi tạo ngẫu nhiên. Vì vậy, sau khi ghi file, không thể xác định điểm nào thuộc nhóm top Reflection Score.

Có thể phát biểu chính xác:

\[
\boxed{
\text{Point cloud mang prior qua spatial sampling, không mang per-point prior}
}
\]

## 6. Point cloud prior có được training dùng không?

Hiện tại không tự động.

Script sinh:

```text
points3d_prior.ply
```

nhưng loader lại đọc:

- COLMAP: `sparse/0/points3D.ply`;
- synthetic: `points3d.ply`.

Không có run script nào tự gọi `generate_prior_pcd.py` rồi nối output vào loader. Vì vậy pipeline thực tế hiện nay vẫn là:

```text
RGB
  ↓
2D RefScore/RefConf
  ↓
Nạp theo camera
  ↓
Hướng dẫn ADC/loss/masking
```

chứ chưa phải:

```text
RGB
  ↓
2D confidence
  ↓
3D multi-view accumulation
  ↓
Prior-aware point cloud
  ↓
Gaussian initialization
```

## Kết luận

| Câu hỏi | Trạng thái hiện tại |
|---|---|
| Có confidence map 2D cho từng ảnh? | Có |
| 2D extractor tự chiếu lên 3D? | Không |
| Repo có script chiếu và tích lũy đa góc nhìn? | Có, riêng cho synthetic |
| Có space carving? | Có, dựa trên alpha silhouette |
| PLY có lưu Reflection Score từng điểm? | Không |
| Point cloud có prior gián tiếp qua mật độ điểm? | Có |
| Training tự động dùng `points3d_prior.ply`? | Không |
| Pipeline mặc định có thêm Gaussian prior khi initialization? | Không |

Muốn biến nó thành prior-aware initialization hoàn chỉnh, cần nối `points3d_prior.ply` vào loader và lưu `ref_score_3d` dưới dạng thuộc tính PLY hoặc sidecar tensor để `GaussianModel.create_from_pcd()` thực sự nhận được confidence của từng điểm.

---

Vậy tức là points3d_prior.ply nếu như được sử dụng thì sẽ phục vụ vai trò là bổ sung thêm mật độ đám mây điểm khởi đầu tại các vùng chứa reflection score cao thôi phải không?

Đúng, nhưng chính xác hơn là `points3d_prior.ply` sẽ thay đổi phân bố point cloud khởi đầu để vùng có accumulated Reflection Score cao được lấy mẫu dày hơn.

Nó không tự động “bổ sung” điểm vào point cloud gốc theo kiểu hợp hai đám mây:

\[
P_{\text{init}}=P_{\text{original}}\cup P_{\text{prior}}
\]

Trong implementation hiện tại, script tạo một point cloud mới với budget tối đa 100.000 điểm:

\[
P_{\text{prior}}
=
P_{\text{base-random}}
\cup
P_{\text{high-refscore}}
\]

Cụ thể:

- 50.000 điểm được lấy ngẫu nhiên từ visual hull để phủ geometry chung;
- 50.000 điểm có accumulated Reflection Score cao nhất để tăng mật độ tại vùng nghi ngờ phản xạ.

Nếu dùng file này làm input initialization, nó sẽ thay thế `points3d.ply`, chứ chưa được merge với point cloud ban đầu.

## Nó cung cấp điều gì?

Chủ yếu là geometric coverage tốt hơn ở vùng phản xạ:

```text
Reflection Score cao
       ↓
Nhiều điểm 3D được chọn tại vùng đó
       ↓
Nhiều Gaussian được khởi tạo tại vùng đó
       ↓
ADC ít phải xây dựng coverage từ đầu
```

Đây là spatial-density prior.

## Nó không cung cấp điều gì?

Các điểm được chọn không mang theo:

- Reflection Score 3D;
- nhãn `specular`;
- confidence;
- ASG activation flag;
- ASG feature đã được khởi tạo đặc biệt;
- màu đúng từ ảnh.

Sau khi point cloud được nạp, tất cả Gaussian vẫn được khởi tạo theo cùng một cơ chế. Cụ thể, ASG feature của mọi Gaussian đều khởi tạo bằng zero:

\[
\mathbf f_{\text{ASG},i}=\mathbf 0
\]

Vì vậy hệ thống không biết Gaussian nào là 50.000 điểm “specular” và điểm nào là base point. Thông tin prior chỉ còn tồn tại trong phân bố không gian của các điểm.

## Một lưu ý quan trọng

Do script dùng visual-hull voxel và alpha carving thay vì depth/surface reconstruction, “mật độ cao tại vùng reflection” chưa chắc đồng nghĩa với mật độ cao chính xác trên bề mặt phản xạ. Nó có thể tạo nhiều điểm dọc theo thể tích visual hull tương ứng với các tia nhìn có Reflection Score cao.

Tóm lại:

\[
\boxed{
\texttt{points3d\\_prior.ply}
\text{ là density-biased initialization, không phải material-aware initialization}
}
\]

Nếu muốn nó trở thành material-aware initialization thực sự, cần lưu thêm `ref_score_3d` cho từng điểm và truyền thuộc tính đó vào `GaussianModel`, chẳng hạn để:

- khởi tạo ASG capacity khác nhau;
- điều khiển ASG gating;
- đặt learning rate hoặc regularization khác nhau;
- bảo vệ/ưu tiên Gaussian specular trong VCD/VCP.

---
Bạn có thể trình bày cơ chế mà visual của các vùng reflection/specular được tối ưu trong pipeline optimization hiện tại của spec-fastgs không?

Trong pipeline optimization hiện tại, visual của vùng reflection/specular được tối ưu chủ yếu bằng reconstruction loss chung của ảnh `SH + ASG`, kết hợp với Reflection Score để tăng geometric coverage và hạn chế nhẹ SH bậc cao tại vùng phản xạ.

Điểm quan trọng nhất là:

> Không có ground-truth specular image hay loss tách riêng diffuse/specular. SH và ASG cùng cạnh tranh để giải thích ảnh RGB đầu vào.

## 1. Luồng tối ưu tổng quát

Với mỗi iteration:

```text
Chọn một training camera
        ↓
Tính view direction và normal từng Gaussian
        ↓
SH → base color
ASG feature + viewdir + normal → Specular MLP → specular RGB
        ↓
Màu Gaussian = SH RGB + ASG RGB
        ↓
Alpha compositing/rasterization
        ↓
So sánh ảnh render với ground truth
        ↓
L1 + SSIM
        ↓
Backpropagate vào:
geometry + opacity + SH + ASG features + Specular MLP
```

Ảnh cuối được biểu diễn:

\[
I_{\text{render}}
=
\mathcal R
\left(
C_{\text{SH}}+C_{\text{ASG}},
\alpha,\mu,\Sigma
\right)
\]

Trong đó \(\mathcal R\) là quá trình Gaussian rasterization và alpha compositing.

## 2. Giai đoạn đầu: SH và geometry học ảnh nền

Theo run script hiện tại:

```bash
--specular_start_iter 3000
```

Vì điều kiện trong code là:

```python
if iteration > opt.specular_start_iter:
```

nên trong khoảng iteration \(1\rightarrow3000\):

\[
C_{\text{ASG}}=0
\]

và:

\[
I_{\text{render}}=\mathcal R(C_{\text{SH}})
\]

Giai đoạn này tối ưu:

- vị trí Gaussian;
- scale và rotation;
- opacity;
- SH DC/base color;
- các hệ số SH bậc cao.

Mục tiêu là tạo một representation nền tương đối ổn định trước khi nhánh specular được kích hoạt.

Tuy nhiên, vì SH degree 3 vẫn có khả năng biểu diễn view-dependent color, trong giai đoạn đầu SH có thể bắt đầu học một phần highlight.

## 3. Sau iteration 3000: nhánh ASG tham gia

Mỗi Gaussian có một vector:

```python
gaussians.get_asg_features
```

ASG feature không phải RGB trực tiếp. Nó được đưa qua Specular MLP cùng với hướng nhìn và normal.

### 3.1 Hướng nhìn

Với Gaussian \(i\) và camera \(c\):

\[
\mathbf v_i
=
\frac{\mathbf x_i-\mathbf c}
{\|\mathbf x_i-\mathbf c\|+\epsilon}
\]

### 3.2 Normal

Normal được lấy từ trục nhỏ nhất của covariance Gaussian:

\[
\mathbf n_i=
\operatorname{MinimumAxis}
(\operatorname{scale}_i,\operatorname{rotation}_i)
\]

Normal được flip để hướng phù hợp với camera.

### 3.3 ASG encoding

Đối với synthetic network, code còn tính reflection direction:

\[
\mathbf r
=
2(\mathbf{-v}\cdot\mathbf n)\mathbf n+\mathbf v
\]

ASG sử dụng các lobe định hướng với amplitude và độ sắc theo hai phương:

\[
G_k(\mathbf r)
=
a_k\,
\max(\mathbf r\cdot\omega_k,0)
\exp
\left[
-\lambda_k(\omega_{\lambda,k}\cdot\mathbf r)^2
-\mu_k(\omega_{\mu,k}\cdot\mathbf r)^2
\right]
\]

Các response này cùng view-direction encoding được đưa qua MLP để sinh:

\[
C_{\text{ASG},i}
=
f_\theta
(\mathbf f_{\text{ASG},i},\mathbf v_i,\mathbf n_i)
\in\mathbb R^3
\]

Do đầu ra phụ thuộc camera direction, một Gaussian có thể phát sáng ở một góc nhìn nhưng gần như không đóng góp specular ở góc khác. Đây là cơ chế trực tiếp tái tạo highlight dịch chuyển theo viewpoint.

## 4. SH và ASG được kết hợp thế nào?

Renderer cộng trực tiếp hai thành phần ở Gaussian space:

\[
C_i=C_{\text{SH},i}+C_{\text{ASG},i}
\]

Sau đó mới rasterize:

```python
colors_precomp = sh_color + mlp_color
```

Không có công thức vật lý kiểu:

\[
C=C_{\text{diffuse}}+F(\theta)C_{\text{reflection}}
\]

với Fresnel, roughness hay metallic riêng biệt. ASG chỉ là một nhánh view-dependent có cấu trúc định hướng, còn MLP học RGB residual phù hợp với ảnh.

## 5. Loss trực tiếp tối ưu visual specular

Loss chính là:

\[
L_{\text{photo}}
=
(1-\lambda_{\text{DSSIM}})L_1
+
\lambda_{\text{DSSIM}}(1-\operatorname{SSIM})
\]

với mặc định:

\[
\lambda_{\text{DSSIM}}=0.2
\]

hay:

\[
L_{\text{photo}}
=
0.8L_1+0.2(1-\operatorname{SSIM})
\]

Gradient từ cùng một loss đi vào:

- SH coefficients;
- ASG feature của từng Gaussian;
- trọng số Specular MLP;
- opacity;
- position, scale và rotation thông qua rasterizer.

Nếu highlight bị thiếu trong ảnh render, loss sẽ tạo gradient làm tăng khả năng biểu diễn của SH hoặc ASG, tùy nhánh nào có thể giảm loss hiệu quả hơn.

## 6. Không có explicit specular supervision

Hiện tại không có:

- ground-truth specular layer;
- ground-truth diffuse layer;
- loss riêng so sánh `only_asg` với specular image;
- constraint bắt ASG bằng 0 tại vùng diffuse;
- constraint buộc SH chỉ chứa diffuse.

Vì vậy decomposition:

\[
I_{\text{GT}}
\approx
I_{\text{SH}}+I_{\text{ASG}}
\]

không duy nhất. Một highlight có thể được học bởi:

- SH;
- ASG;
- cả hai;
- hoặc geometry/opacity bị điều chỉnh để bù lỗi.

Đây là lý do repo có thêm Reflection Score và SH-spec mask để tạo inductive bias cho sự phân vai.

## 7. Reflection Score cải thiện visual theo ba đường

### 7.1 Tăng geometric coverage ở vùng reflection

Trong VCD/ADC:

\[
M_{\text{ADC}}
=
M_{\text{SH-error}}\lor M_{\text{RefScore}}
\]

Gaussian chiếu vào vùng RefScore cao dễ đạt multi-view importance condition hơn và có cơ hội clone/split cao hơn.

Kết quả gián tiếp:

```text
Vùng highlight/reflection
        ↓
Nhiều Gaussian hơn
        ↓
Nhiều spatial samples và ASG features hơn
        ↓
Khả năng tái tạo biên highlight và chi tiết specular tốt hơn
```

Reflection Score ở đây không trực tiếp bảo ASG phải sinh màu gì. Nó cấp thêm geometric/representation capacity.

### 7.2 Adaptive Reflection Score

Các run script hiện tại bật:

```bash
USE_ADAPTIVE_PRIOR=True
```

Theo chu kỳ, pipeline render ảnh base/SH-only:

\[
I_{\text{base}}=\mathcal R(C_{\text{SH}})
\]

rồi tính residual:

\[
R(x,y)
=
\operatorname{mean}_{RGB}
|I_{\text{base}}(x,y)-I_{\text{GT}}(x,y)|
\]

Residual được chuẩn hóa bằng phân vị 95% và nhân với static prior:

\[
A(x,y)
=
R_{\text{norm}}(x,y)M_{\text{static}}(x,y)
\]

Sau đó cập nhật EMA:

\[
M_t
=
\beta M_{t-1}
+(1-\beta)A
\]

với mặc định:

\[
\beta=0.7
\]

Ý nghĩa:

- vùng heuristic cho là reflection;
- đồng thời SH-only vẫn chưa giải thích được;
- sẽ tiếp tục giữ Reflection Score cao.

Prior vì thế tập trung dần vào nơi base representation còn thiếu.

### 7.3 Giảm khả năng SH bậc cao chiếm highlight

Các run script hiện tại bật:

```bash
USE_SH_SPEC_MASK=True
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MASK_THRESHOLD=0.75
```

Pipeline threshold confidence map:

\[
M_{\text{spec}}(x,y)
=
[RefConf(x,y)>0.75]
\]

Rasterizer chiếu mask pixel về Gaussian. Gaussian phải đóng góp vào đủ số pixel/view count quy định mới được đánh dấu.

Với Gaussian nằm trong vùng specular, gradient của SH bậc cao `_features_rest` được scale:

\[
\nabla f_{\text{SH-rest},i}
\leftarrow
0.75\,\nabla f_{\text{SH-rest},i}
\]

Điều này không đóng băng SH mà chỉ làm nó học chậm hơn 25% tại vùng reflection có confidence cao.

Trong khi đó:

- SH DC/base color vẫn học bình thường;
- ASG feature vẫn nhận full gradient;
- Specular MLP vẫn nhận full gradient.

Kết quả mong muốn:

```text
SH DC → màu nền ổn định
SH bậc cao → vẫn học nhưng yếu hơn ở vùng specular
ASG → có lợi thế tương đối để học view-dependent highlight
```

## 8. ASG feature và Specular MLP được tối ưu riêng

Có hai nhóm tham số specular:

### Per-Gaussian ASG feature

\[
\mathbf f_{\text{ASG},i}
\]

Mỗi Gaussian có feature riêng và được tối ưu bằng `asg_optimizer`.

### Shared Specular MLP

\[
f_\theta
\]

MLP dùng chung cho tất cả Gaussian và được tối ưu bằng một Adam optimizer riêng với learning-rate scheduler.

Do đó mô hình học hai loại thông tin:

- ASG feature: đặc tính specular cục bộ của từng Gaussian;
- MLP: quy luật chung chuyển ASG encoding và view direction thành RGB.

Cả hai được update từ reconstruction loss, không cần nhãn specular riêng.

## 9. Sparse ASG evaluation hiện tại

Để giảm chi phí, pipeline không luôn chạy Specular MLP trên toàn bộ Gaussian.

Nó dùng visibility mask của iteration trước:

```python
vis_indices = prev_vis_mask.nonzero(...)
```

và chỉ tính ASG cho tập này. Sau đó scatter về buffer toàn cảnh:

```python
mlp_color = zeros(N,3)
mlp_color[vis_indices] = spec_sparse
```

Các Gaussian không nằm trong mask nhận:

\[
C_{\text{ASG}}=0
\]

tại iteration đó.

Khi:

- iteration đầu tiên bật ASG;
- số Gaussian vừa thay đổi sau densification;
- hoặc đến `full_asg_interval`;

pipeline có thể đánh giá toàn bộ Gaussian.

Tuy nhiên cấu hình hiện tại có:

```python
full_asg_interval = 0
```

nên không có full refresh định kỳ; full evaluation chủ yếu xảy ra khi chưa có compatible previous mask hoặc số Gaussian thay đổi.

Đây là tối ưu tốc độ, nhưng có một điểm cần lưu ý: camera được lấy ngẫu nhiên và camera iteration hiện tại thường khác camera trước. Dùng previous-camera visibility có thể bỏ qua một số Gaussian đang nhìn thấy ở camera hiện tại nhưng không nhìn thấy ở camera trước. Khi đó chúng tạm thời không có ASG color và không nhận ASG gradient ở iteration đó.

## 10. Weighted specular loss hiện có nhưng mặc định chưa hoạt động

Code hỗ trợ tăng trọng số L1 tại vùng RefScore cao:

\[
w(x,y)=1+\lambda_{\text{spec-L1}}RefConf(x,y)
\]

\[
L_{1,\text{weighted}}
=
\frac{
\sum_{x,y,c}w(x,y)|I-I_{\text{GT}}|
}{
3\sum_{x,y}w(x,y)
}
\]

Nhưng mặc định:

```python
lambda_spec_l1_weight = 0.0
```

và các run script đang xét không truyền giá trị khác. Do đó cơ chế này hiện không tham gia optimization.

Nói cách khác, Reflection Score không trực tiếp tăng photometric penalty tại highlight trong cấu hình hiện tại.

## 11. Specular regularization cũng mặc định tắt

Code có:

\[
L_{\text{spec-reg}}
=
\|C_{\text{ASG}}\|_2^2
\]

và:

\[
L=L_{\text{photo}}
+\lambda_{\text{spec-reg}}L_{\text{spec-reg}}
\]

nhằm ngăn ASG energy tăng không kiểm soát hoặc tràn sang vùng diffuse.

Nhưng mặc định:

```python
lambda_spec_reg = 0.0
```

nên ASG hiện không bị regularize trực tiếp.

## 12. Normal có được ASG tối ưu không?

Trong lời gọi Specular MLP:

```python
normal.detach()
```

Do đó gradient từ nhánh ASG không truyền ngược qua normal:

\[
\frac{\partial C_{\text{ASG}}}{\partial \mathbf n}
\not\rightarrow
\text{scale/rotation}
\]

ASG dùng normal để dự đoán highlight, nhưng không trực tiếp xoay Gaussian nhằm cải thiện normal thông qua nhánh specular.

Geometry vẫn nhận gradient từ:

- rasterization;
- alpha compositing;
- ảnh tổng SH+ASG;
- vị trí và covariance ảnh hưởng footprint.

Điều này giúp hạn chế ASG làm méo geometry, nhưng normal specular phụ thuộc vào geometry học được từ các tín hiệu khác.

## 13. Tóm tắt theo từng giai đoạn

### Iteration 1–3000

```text
SH-only render
→ học geometry, opacity và base appearance
→ Reflection Score có thể hướng dẫn ADC từ các kỳ densification phù hợp
```

### Iteration 3001–8000

```text
SH + ASG render
→ ASG feature và Specular MLP bắt đầu học highlight từ reconstruction loss
→ SH vẫn học tự do tại mọi vùng
→ adaptive prior cập nhật theo SH-only residual
```

### Từ iteration 8000

```text
SH + ASG tiếp tục cùng tối ưu
→ RefConf được chiếu về Gaussian
→ gradient SH bậc cao ở vùng specular nhân 0.75
→ ASG có lợi thế tương đối để nhận phần view-dependent
```

### Trong densification đến iteration 15.000

```text
RefScore + SH-only residual hướng dẫn VCD
→ tăng Gaussian coverage ở vùng nghi ngờ reflection
→ Gaussian mới kế thừa ASG feature từ Gaussian cha
```

### Sau iteration 15.000

Densification dừng; các lần VCP/final pruning tiếp tục tinh gọn representation. Hiện scoring vẫn dùng SH-only như đã phân tích trước đó.

## Nhận xét tổng thể

Cơ chế hiện tại tối ưu visual reflection qua ba trục:

\[
\boxed{
\text{Representation}
+
\text{Geometric coverage}
+
\text{Role separation}
}
\]

Cụ thể:

1. ASG + Specular MLP cung cấp representation view-dependent.
2. Reflection Score hướng dẫn ADC đặt thêm Gaussian capacity ở vùng phản xạ.
3. SH-spec mask giảm nhẹ gradient SH bậc cao để ASG có cơ hội học highlight.

Tuy nhiên, sự phân rã SH/ASG vẫn là soft disentanglement. Không có loss nào đảm bảo:

\[
C_{\text{SH}}=C_{\text{diffuse}},
\qquad
C_{\text{ASG}}=C_{\text{specular}}
\]

Với cấu hình hiện tại, ASG học specular chủ yếu vì:

- nó phụ thuộc view direction phù hợp với bản chất highlight;
- SH bậc cao bị giảm gradient nhẹ ở vùng RefScore cao;
- vùng reflection được cấp thêm Gaussian thông qua ADC.

Chứ không phải vì hệ thống có supervision specular riêng.

---

"Trong nghiên cứu này, điểm số phản xạ được xây dựng thông qua ba giai
đoạn. Trước tiên, các mô hình heuristic quang học được áp dụng trên từng
ảnh để tạo bản đồ phản xạ hai chiều. Tiếp theo, các bản đồ này được ánh xạ
sang không gian ba chiều thông qua tham số camera và cơ chế khắc không
gian. Cuối cùng, thông tin từ nhiều góc nhìn được tích lũy để tạo điểm số
phản xạ cho từng điểm ba chiều."
Mô tả như này có đúng về bước "Trích xuất điểm số phản xạ (Reflection Score)" không?

Mô tả này đúng về mặt ý tưởng của pipeline mở rộng 2D→3D, nhưng chưa đúng hoàn toàn với pipeline thực tế đang được Spec-FastGS sử dụng.

Cụ thể:

- Giai đoạn 1 đúng: mỗi ảnh được xử lý bằng heuristic để tạo `ref_score` và `ref_conf` 2D.
- Giai đoạn 2–3 có tồn tại trong `generate_prior_pcd.py`, nhưng là script riêng, chỉ hỗ trợ synthetic dataset có alpha mask.
- `extract_reflection_prior.py` tự nó không thực hiện ánh xạ 3D hay tích lũy đa góc nhìn.
- Điểm phản xạ 3D chỉ được tính tạm thời để chọn các điểm top-k.
- `points3d_prior.ply` không lưu Reflection Score của từng điểm.
- Point cloud prior chưa được loader training sử dụng tự động.
- Với dữ liệu real/COLMAP, bước space carving hiện bị bỏ qua.

Do đó, nếu đoạn này nhằm mô tả “phương pháp được đề xuất về mặt lý thuyết”, nó khá phù hợp. Nhưng nếu nhằm mô tả implementation/pipeline huấn luyện hiện tại, nó sẽ khiến người đọc hiểu nhầm rằng cả ba bước đều đang hoạt động tự động.

## Cách viết đúng với implementation hiện tại

> Trong triển khai hiện tại, điểm số phản xạ trước hết được trích xuất độc lập trên từng ảnh huấn luyện bằng các mô hình heuristic quang học, tạo thành các bản đồ Reflection Score và Reflection Confidence hai chiều. Các bản đồ này được nạp theo từng camera và chủ yếu được sử dụng để hướng dẫn cơ chế kiểm soát mật độ Gaussian trong quá trình huấn luyện.  
>
> Đối với dữ liệu tổng hợp có thông tin camera và alpha mask, hệ thống còn cung cấp một bước tiền xử lý ba chiều tùy chọn. Bước này tạo một lưới điểm ba chiều, chiếu các điểm lên từng ảnh, loại bỏ các điểm không phù hợp với silhouette thông qua space carving, sau đó cộng dồn Reflection Score từ các góc nhìn hợp lệ. Điểm tích lũy được dùng để ưu tiên lựa chọn các vị trí có khả năng phản xạ cao khi xây dựng point cloud khởi tạo. Tuy nhiên, điểm số phản xạ ba chiều hiện chỉ được sử dụng trong quá trình chọn mẫu và không được lưu dưới dạng thuộc tính của từng điểm; point cloud sinh ra cũng chưa được kết nối tự động với training loader.

## Nếu muốn giữ cấu trúc ba giai đoạn

Bạn có thể sửa thành:

> Về mặt thiết kế, quá trình xây dựng tiền nghiệm phản xạ ba chiều gồm ba giai đoạn. Trước tiên, các mô hình heuristic quang học được áp dụng độc lập trên từng ảnh để tạo bản đồ điểm số phản xạ hai chiều. Tiếp theo, đối với dữ liệu tổng hợp có alpha mask, một lưới điểm ba chiều được chiếu lên các ảnh bằng tham số camera và được lọc theo visual hull thông qua space carving. Cuối cùng, điểm số hai chiều tại các hình chiếu hợp lệ được cộng dồn qua nhiều góc nhìn để xếp hạng các điểm ba chiều. Các điểm có điểm tích lũy cao được ưu tiên khi tạo point cloud khởi tạo, qua đó tăng mật độ lấy mẫu tại những vùng có khả năng phản xạ cao.

Cách diễn đạt này chính xác hơn vì dùng:

- “tiền nghiệm phản xạ ba chiều” thay cho khẳng định toàn bộ đó là extraction đang dùng;
- “đối với dữ liệu tổng hợp” để nêu phạm vi;
- “lọc theo visual hull” thay vì ngụ ý visibility/depth chính xác;
- “xếp hạng điểm” thay vì nói mỗi điểm mang Reflection Score lâu dài;
- “tăng mật độ lấy mẫu” thay vì ngụ ý Gaussian được gán nhãn specular.

## Một điểm cần chỉnh về thuật ngữ

Câu:

> “tạo điểm số phản xạ cho từng điểm ba chiều”

chỉ đúng trong bộ nhớ khi `generate_prior_pcd.py` đang chạy. Sau khi xuất PLY, score bị loại bỏ. Vì vậy nên viết:

> “tính điểm số phản xạ tích lũy tạm thời cho từng điểm ứng viên ba chiều nhằm phục vụ xếp hạng và chọn mẫu.”

Tóm lại:

\[
\boxed{
\text{Đúng với thiết kế 2D→3D tùy chọn, nhưng không đúng nếu mô tả pipeline training hiện tại}
}
\]

Pipeline training hiện tại chủ yếu là:

\[
\text{RGB}
\rightarrow
\text{2D RefScore/RefConf}
\rightarrow
\text{ADC và các cơ chế optimization}
\]

chứ chưa tự động hoàn thành chuỗi:

\[
\text{2D prior}
\rightarrow
\text{3D accumulated score}
\rightarrow
\text{prior-aware initialization}.
\]
---

Mình không khuyến nghị xóa hẳn file ngay, nhưng nên loại `generate_prior_pcd.py` khỏi pipeline chính và khỏi phần đóng góp cốt lõi của luận văn.

Lý do không chỉ vì nó không hỗ trợ Mip-NeRF 360, mà còn vì cơ chế này hiện chưa thực sự được tích hợp hoặc kiểm chứng:

- Chỉ hỗ trợ synthetic dataset có `transforms_train.json` và alpha mask.
- Không hoạt động trên dữ liệu real/COLMAP.
- `points3d_prior.ply` không được loader tự động sử dụng.
- Các run `toaster` trong `run_note.md` đều dùng `points3d.ply` 100.000 điểm, không dùng prior PLY 66.292 điểm.
- Reflection Score 3D chỉ được dùng để chọn điểm rồi bị loại bỏ, không được lưu thành thuộc tính.
- Space carving theo silhouette không xác định đúng bề mặt nhìn thấy.
- Không có ablation chứng minh prior-guided initialization cải thiện kết quả.
- Màu của point cloud prior được khởi tạo ngẫu nhiên, có thể còn làm initialization kém ổn định.
- Nó thay thế point cloud khởi đầu thay vì bổ sung có kiểm soát vào point cloud gốc.

## Hướng xử lý mình đề xuất

### 1. Giữ file như một thử nghiệm legacy

Di chuyển hoặc đổi tên thành dạng:

```text
experiment/generate_prior_pcd_experimental.py
```

hoặc:

```text
legacy/generate_prior_pcd.py
```

Thêm cảnh báo ở đầu file:

```python
"""
EXPERIMENTAL / NOT USED BY THE CURRENT TRAINING PIPELINE.

This script only supports synthetic Blender-style datasets with alpha masks.
The generated points3d_prior.ply is not automatically loaded by train.py.
"""
```

Cách này giữ lại lịch sử nghiên cứu và tránh mất code, nhưng không khiến người đọc tưởng đây là thành phần production.

### 2. Loại nó khỏi mô tả Reflection Score chính

Pipeline chính nên được trình bày là:

\[
\text{RGB images}
\rightarrow
\text{2D Reflection Score/Confidence}
\rightarrow
\text{camera-conditioned training prior}
\rightarrow
\text{ADC + role separation + optional weighted loss}
\]

Không nên mô tả mặc định là:

\[
\text{2D prior}
\rightarrow
\text{space carving}
\rightarrow
\text{3D accumulated prior}
\rightarrow
\text{prior-guided initialization}
\]

vì chuỗi thứ hai không được sử dụng trong các thực nghiệm hiện tại.

### 3. Định vị đóng góp Reflection Score đúng bản chất

Đóng góp có bằng chứng thực nghiệm của repo hiện tại là:

> Reflection Score là một tiền nghiệm ảnh 2D theo từng camera, được sử dụng để hướng dẫn geometric coverage trong ADC và hỗ trợ phân vai SH–ASG.

Cách mô tả này phù hợp với cả:

- Mip-NeRF 360;
- Ref-NeRF synthetic;
- các run trong `run_note.md`;
- implementation của `compute_gaussian_score_fastgs()`.

## Vì sao không nên cố giữ nó trong luận văn?

Nếu giữ cơ chế 3D prior như một phần chính, người phản biện có thể hỏi:

- Kết quả nào dùng `points3d_prior.ply`?
- Ablation giữa point cloud gốc và prior point cloud ở đâu?
- Vì sao Mip-NeRF 360 không dùng được?
- Score 3D được lưu vào Gaussian ở trường nào?
- Làm sao xử lý occlusion khi chỉ có silhouette?
- Tại sao gọi là point-wise reflection prior khi PLY không lưu score?
- Vì sao initial Gaussian trong log vẫn là 100.000?

Với code và log hiện tại, các câu hỏi này khó trả lời thuyết phục. Loại nó khỏi phương pháp chính sẽ làm luận văn nhất quán hơn.

## Có nên xóa hoàn toàn không?

Chỉ nên xóa hoàn toàn nếu bạn chắc chắn không cần:

- lưu lịch sử thử nghiệm;
- trình bày hướng nghiên cứu thất bại;
- phát triển lại 3D prior sau này.

Còn hiện tại, phương án tốt nhất là:

\[
\boxed{
\text{Giữ code, nhưng hạ xuống trạng thái experimental/unused}
}
\]

Đồng thời:

- bỏ lời gọi hoặc hướng dẫn `generate_prior_pcd.py` khỏi README chính;
- sửa tài liệu ba giai đoạn thành pipeline 2D;
- ghi rõ `points3d_prior.ply` không được dùng trong kết quả báo cáo;
- không gọi cơ chế hiện tại là “multi-view 3D Reflection Score” trong phần phương pháp chính.

## Nếu muốn phục hồi ý tưởng trong tương lai

Một phiên bản 3D prior đáng giữ làm đóng góp sẽ cần:

1. Hỗ trợ camera COLMAP/Mip-NeRF 360.
2. Chiếu COLMAP points hoặc reconstructed surface thay vì voxel visual hull.
3. Xử lý visibility bằng depth/z-buffer.
4. Chuẩn hóa score theo số observation.
5. Lưu `ref_score_3d` cho từng điểm.
6. Nối trực tiếp vào dataset loader và `GaussianModel`.
7. Có ablation initialization gốc so với prior initialization.
8. Chứng minh cải thiện cả synthetic và real dataset.

Ở thời điểm hiện tại, tập trung Reflection Score vào ADC là lựa chọn gọn và có cơ sở thực nghiệm hơn.
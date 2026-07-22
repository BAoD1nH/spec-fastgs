# Tối ưu biểu diễn SH–ASG

## Algorithm 5: Hybrid SH–ASG Forward Rendering

**Input**

- Mô hình Gaussian \(G\).
- Mạng specular \(F_{\mathrm{spec}}\).
- Camera \(C_c\).
- Iteration \(t\).
- Visibility mask từ iteration trước \(V_{\mathrm{prev}}\).

**Output**

- Ảnh render \(I_{\mathrm{render}}\).
- Metadata rasterization \(R\).

```text
01  FOR EACH Gaussian g_i ∈ G DO
02      v_i ← NORMALIZE(μ_i - CAMERA_CENTER(C_c))
03      n_i ← MINIMUM_COVARIANCE_AXIS(s_i, r_i)
04      n_i ← FLIP_NORMAL_TOWARD_VIEW(n_i, v_i)
05      C_SH,i ← EVALUATE_SPHERICAL_HARMONICS(f_SH,i, v_i)
06  END FOR
07
08  C_ASG ← ZERO_RGB_ARRAY(|G|)
09
10  IF t > t_spec-start THEN
11      IF V_prev IS VALID FOR |G| THEN
12          V_eval ← INDICES_SELECTED_BY(V_prev)
13      ELSE
14          V_eval ← {1, ..., |G|}
15      END IF
16
17      FOR EACH i ∈ V_eval DO
18          C_ASG,i ← F_spec(f_ASG,i, v_i, STOP_GRADIENT(n_i))
19      END FOR
20  END IF
21
22  FOR EACH Gaussian g_i ∈ G DO
23      C_i ← C_SH,i + C_ASG,i
24  END FOR
25
26  I_render, R ← GAUSSIAN_RASTERIZE(
27      positions = {μ_i},
28      covariances = {Σ_i},
29      opacities = {α_i},
30      colors = {C_i},
31      camera = C_c
32  )
33
34  RETURN I_render, R
```

## Algorithm 6: Tính hàm mất mát huấn luyện

**Input**

- Ảnh render \(I_{\mathrm{render}}\) và ground truth \(I_{\mathrm{GT}}\).
- Reflection Confidence map \(M_{\mathrm{conf}}\).
- Trọng số reflection \(\lambda_{\mathrm{ref}}\).
- Trọng số DSSIM \(\lambda_{\mathrm{DSSIM}}\).
- Trọng số ASG regularization \(\lambda_{\mathrm{reg}}\).
- ASG RGB output \(C_{\mathrm{ASG}}\).

**Output**

- Training loss \(L\).

```text
01  E_L1 ← ABS(I_render - I_GT)
02
03  IF λ_ref > 0 AND M_conf IS AVAILABLE THEN
04      W ← 1 + λ_ref × M_conf
05      L_L1 ← SUM(W × E_L1) / (3 × SUM(W) + ε)
06  ELSE
07      L_L1 ← MEAN(E_L1)
08  END IF
09
10  L_SSIM  ← 1 - SSIM(I_render, I_GT)
11  L_photo ← (1 - λ_DSSIM) × L_L1 + λ_DSSIM × L_SSIM
12
13  IF λ_reg > 0 AND C_ASG IS AVAILABLE THEN
14      L_reg ← MEAN(C_ASG²)
15  ELSE
16      L_reg ← 0
17  END IF
18
19  L ← L_photo + λ_reg × L_reg
20  RETURN L
```

## Algorithm 7: Phân vai SH–ASG tại vùng phản xạ

**Input**

- Reflection Confidence map \(M_{\mathrm{conf},c}\).
- Mô hình Gaussian \(G\) và camera \(C_c\).
- Ngưỡng confidence \(\tau_{\mathrm{spec}}\).
- Số lần đóng góp tối thiểu \(K\).
- Hệ số gradient SH \(\beta\).
- Iteration \(t\).

**Output**

- Gradient SH bậc cao đã được điều chỉnh.

```text
01  IF role separation IS DISABLED OR t < t_SH-mask-start THEN
02      RETURN
03  END IF
04
05  M_spec ← INDICATOR(M_conf,c > τ_spec)
06  count ← RASTERIZE_AND_ACCUMULATE_MASK(
07      G, camera = C_c, metric_map = M_spec
08  )
09
10  FOR EACH Gaussian g_i ∈ G DO
11      Q_i ← INDICATOR(count_i ≥ K)
12      IF Q_i = 1 THEN
13          ∇f_SH-rest,i ← β × ∇f_SH-rest,i
14      END IF
15  END FOR
16
17  RETURN modified SH gradients
```

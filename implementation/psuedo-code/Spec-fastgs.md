# Thuật toán tổng quát Spec-FastGS

## Algorithm 1: Spec-FastGS Training Pipeline

**Input**

- Tập ảnh huấn luyện đa góc nhìn \(\mathcal{I}=\{I_c\}\).
- Tập camera \(\mathcal{C}=\{C_c\}\).
- Point cloud khởi tạo \(P\).
- Số iteration \(T\).
- Cấu hình reflection prior \(\Theta_{\mathrm{ref}}\).
- Cấu hình Adaptive Density Control \(\Theta_{\mathrm{ADC}}\).
- Cấu hình biểu diễn SH–ASG \(\Theta_{\mathrm{spec}}\).

**Output**

- Mô hình Gaussian đã tối ưu \(G^*\).
- Mạng specular đã tối ưu \(F_{\mathrm{spec}}^*\).

```text
01  // Stage 1–2: Offline reflection-prior extraction
02  FOR EACH training image I_c ∈ ℐ DO
03      M_score,c ← EXTRACT_2D_REFLECTION_SCORE(I_c, Θ_ref)
04      M_conf,c  ← BUILD_REFLECTION_CONFIDENCE(M_score,c, Θ_ref)
05      SAVE(M_score,c, M_conf,c)
06  END FOR
07
08  // Stage 3: Standard Gaussian initialization
09  G      ← INITIALIZE_GAUSSIANS(P)
10  F_spec ← INITIALIZE_SPECULAR_NETWORK(Θ_spec)
11  ATTACH_CAMERA_PRIORS(C, {M_score,c}, {M_conf,c})
12
13  FOR t ← 1 TO T DO
14      c ← SAMPLE_TRAINING_CAMERA(C)
15
16      // Stage 4: Hybrid SH–ASG forward rendering
17      C_SH ← EVALUATE_SH_COLOR(G, C_c)
18      IF t > t_spec-start THEN
19          V_eval ← SELECT_ASG_EVALUATION_SET(G)
20          C_ASG  ← EVALUATE_ASG_COLOR(G[V_eval], F_spec, C_c)
21      ELSE
22          C_ASG ← 0
23      END IF
24      I_render, R ← RASTERIZE(G, C_SH + C_ASG, C_c)
25
26      // Stage 5–6: Loss and SH–ASG role separation
27      L ← COMPUTE_TRAINING_LOSS(I_render, I_c, M_conf,c, Θ_spec)
28      BACKPROPAGATE(L)
29      IF SH_SPEC_ROLE_SEPARATION_IS_ACTIVE(t) THEN
30          Q ← PROJECT_REFLECTION_MASK_TO_GAUSSIANS(M_conf,c, G, C_c)
31          SCALE_HIGH_ORDER_SH_GRADIENT(G, Q, Θ_spec)
32      END IF
33
34      UPDATE_GAUSSIAN_PARAMETERS(G)
35      UPDATE_ASG_FEATURES(G)
36      UPDATE_SPECULAR_NETWORK(F_spec)
37
38      // Stage 7: Optional residual-adaptive prior
39      IF SHOULD_UPDATE_ADAPTIVE_PRIOR(t) THEN
40          UPDATE_ADAPTIVE_REFLECTION_PRIORS(
41              G, C, {M_score,c}, {M_conf,c}, Θ_ref
42          )
43      END IF
44
45      // Stage 8–9: Adaptive Density Control
46      IF SHOULD_RUN_ADC(t) THEN
47          C_score ← SAMPLE_SCORING_CAMERAS(C)
48          S_dense, S_prune ← COMPUTE_MULTIVIEW_SCORES(
49              G, C_score, {M_score,c}, Θ_ADC
50          )
51          G ← VCD_DENSIFY(G, S_dense, Θ_ADC)
52          G ← VCP_PRUNE(G, S_prune, Θ_ADC)
53      END IF
54
55      IF SHOULD_RUN_FINAL_PRUNING(t) THEN
56          S_prune ← COMPUTE_MULTIVIEW_PRUNING_SCORE(G, C)
57          G ← FINAL_VCP_PRUNE(G, S_prune)
58      END IF
59  END FOR
60
61  SAVE(G, F_spec)
62  RETURN G, F_spec
```

> **Phạm vi:** Thuật toán trên mô tả pipeline huấn luyện đang được sử dụng. `generate_prior_pcd.py` không thuộc pipeline chính này.

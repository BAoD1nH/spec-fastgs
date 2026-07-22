# Adaptive Prior, VCD và VCP

## Algorithm 8: Residual-Adaptive Reflection Prior

**Input**

- Static priors \(\{M_{\mathrm{static},c}\}\).
- Current priors \(\{M_{t,c}\}\).
- Mô hình Gaussian \(G\).
- Tập camera cập nhật \(\mathcal{C}_{\mathrm{adapt}}\).
- Hệ số EMA \(\beta\).

**Output**

- Tập prior đã cập nhật \(\{M_{t+1,c}\}\).

```text
01  FOR EACH camera C_c ∈ C_adapt DO
02      I_SH ← RENDER(G, camera = C_c, ASG_color = 0)
03      R_c  ← MEAN_RGB(ABS(I_SH - I_GT,c))
04      r_95 ← QUANTILE(R_c, 0.95)
05      R_norm,c ← CLAMP(R_c / (r_95 + ε), 0, 1)
06
07      M_adapt,c ← R_norm,c × M_static,c
08      M_adapt,c ← NORMALIZE_TO_UNIT_INTERVAL(M_adapt,c)
09      M_t+1,c ← β M_t,c + (1 - β) M_adapt,c
10
11      UPDATE_CONFIDENCE_MAP_WITH_SAME_EMA_RULE(c)
12  END FOR
13
14  RETURN {M_t+1,c}
```

## Algorithm 9: VCD scoring và densification hiện tại

**Input**

- Mô hình Gaussian \(G\).
- Tập scoring camera \(\mathcal{C}_{\mathrm{score}}\).
- Reflection Score maps \(\{M_{\mathrm{score},c}\}\).
- Iteration \(t\) và cấu hình \(\Theta_{\mathrm{ADC}}\).

**Output**

- Mô hình Gaussian đã densify \(G'\).

```text
01  S_count ← ZERO_VECTOR(|G|)
02
03  FOR EACH camera C_c ∈ C_score DO
04      I_SH ← RENDER(G, camera = C_c, ASG_color = 0)
05      E_c  ← NORMALIZE(MEAN_RGB(ABS(I_SH - I_GT,c)))
06      M_error,c ← INDICATOR(E_c > τ_loss)
07
08      IF REF_SCORE_IS_ACTIVE(t, |G|) THEN
09          τ_ref ← COMPUTE_DYNAMIC_REFSCORE_THRESHOLD(t, |G|)
10          M_ref,c ← INDICATOR(M_score,c > τ_ref)
11          M_metric,c ← M_error,c OR M_ref,c
12      ELSE
13          M_metric,c ← M_error,c
14      END IF
15
16      count_c ← RASTERIZE_AND_ACCUMULATE_MASK(
17          G, camera = C_c, metric_map = M_metric,c
18      )
19      S_count ← S_count + count_c
20  END FOR
21
22  S_dense    ← FLOOR(S_count / |C_score|)
23  M_multiview ← INDICATOR(S_dense > τ_multiview)
24
25  FOR EACH Gaussian g_i ∈ G DO
26      Q_clone,i ← (||∇μ_i|| ≥ τ_grad) AND (MAX(s_i) ≤ τ_scale)
27      Q_split,i ← (||∇_abs μ_i|| ≥ τ_grad-abs) AND (MAX(s_i) > τ_scale)
28
29      IF M_multiview,i = 1 AND Q_clone,i THEN
30          G ← CLONE_GAUSSIAN(G, g_i)
31      END IF
32      IF M_multiview,i = 1 AND Q_split,i THEN
33          G ← SPLIT_GAUSSIAN(G, g_i)
34      END IF
35  END FOR
36
37  RETURN G
```

## Algorithm 10: VCP scoring và pruning hiện tại

**Input**

- Mô hình Gaussian \(G\).
- Tập scoring camera \(\mathcal{C}_{\mathrm{score}}\).
- Ngưỡng opacity \(\tau_{\alpha}\), size và final-pruning \(\tau_{\mathrm{prune}}\).

**Output**

- Mô hình Gaussian đã prune \(G'\).

```text
01  S_weighted ← ZERO_VECTOR(|G|)
02
03  FOR EACH camera C_c ∈ C_score DO
04      I_SH ← RENDER(G, camera = C_c, ASG_color = 0)
05      L_photo,c ← PHOTOMETRIC_LOSS(I_SH, I_GT,c)
06      M_error,c ← HIGH_ERROR_PIXEL_MASK(I_SH, I_GT,c)
07      count_c ← RASTERIZE_AND_ACCUMULATE_MASK(
08          G, camera = C_c, metric_map = M_error,c
09      )
10      S_weighted ← S_weighted + L_photo,c × count_c
11  END FOR
12
13  S_prune ← MIN_MAX_NORMALIZE(S_weighted)
14  M_low-opacity ← INDICATOR(α < τ_α)
15  M_too-large   ← SCREEN_SIZE_TOO_LARGE OR WORLD_SIZE_TOO_LARGE
16
17  IF regular ADC pruning THEN
18      M_candidate ← M_low-opacity OR M_too-large
19      G ← REMOVE_SELECTED_SUBSET(G, M_candidate, S_prune)
20  ELSE IF final pruning THEN
21      M_bad-score ← INDICATOR(S_prune > τ_prune)
22      M_final ← M_low-opacity OR M_bad-score
23      G ← REMOVE_GAUSSIANS(G, M_final)
24  END IF
25
26  RETURN G
```

## Algorithm 11: ASG-aware VCD/VCP đề xuất

**Trạng thái:** Thiết kế đề xuất, chưa phải implementation hiện tại.

**Input**

- Mô hình Gaussian \(G\) và mạng specular \(F_{\mathrm{spec}}\).
- Tập scoring camera \(\mathcal{C}_{\mathrm{score}}\).
- Reflection priors \(\{M_{\mathrm{score},c}\}\).
- Iteration \(t\).

**Output**

- Mô hình Gaussian đã densify/prune \(G'\).

```text
01  INITIALIZE per-Gaussian VCD and VCP scores
02
03  FOR EACH camera C_c ∈ C_score DO
04      I_SH   ← RENDER_SH_ONLY(G, C_c)
05      I_full ← RENDER_SH_PLUS_ASG(G, F_spec, C_c)
06      E_SH   ← PIXEL_ERROR(I_SH, I_GT,c)
07      E_full ← PIXEL_ERROR(I_full, I_GT,c)
08      M_ref  ← INDICATOR(M_score,c > τ_ref)
09
10      IF t < t_ASG-stable THEN
11          M_VCD ← INDICATOR(E_SH > τ_SH) OR M_ref
12      ELSE
13          M_capacity   ← INDICATOR(E_SH > τ_SH) OR M_ref
14          M_unresolved ← INDICATOR(E_full > τ_full)
15          M_VCD ← M_unresolved AND M_capacity
16      END IF
17      ACCUMULATE_VCD_SCORE(G, C_c, M_VCD)
18
19      M_VCP ← BUILD_ERROR_MASK(E_full)
20      ACCUMULATE_VCP_SCORE(G, C_c, M_VCP)
21  END FOR
22
23  G ← DENSIFY_IF(
24      G, multiview_VCD_consensus AND position_gradient_condition
25  )
26  G ← PRUNE_IF(
27      G, low_opacity OR (poor_full_consistency AND poor_visibility)
28  )
29
30  RETURN G
```

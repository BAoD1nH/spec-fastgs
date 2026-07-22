# Reflection Score và Reflection Confidence

## Algorithm 2: Trích xuất Reflection Score 2D

**Input**

- Ảnh RGB \(I\in[0,1]^{H\times W\times3}\).
- Phương pháp \(m\in\{\text{Tan},\text{Shafer},\text{Hybrid}\}\).
- Tập ngưỡng heuristic \(\Theta_{\mathrm{ref}}\).

**Output**

- Reflection Score map \(M_{\mathrm{score}}\in[0,1]^{H\times W}\).

```text
01  I_max ← CHANNEL_MAX(I)
02  I_min ← CHANNEL_MIN(I)
03  S     ← 1 - I_min / (I_max + ε)          // saturation
04
05  SWITCH m DO
06      CASE Tan:
07          M_raw ← ZERO_MAP(H, W)
08          FOR EACH pixel p DO
09              IF I_min(p) > τ_min AND I_max(p) > τ_bright THEN
10                  M_raw(p) ← I_min(p)
11              END IF
12          END FOR
13
14      CASE Shafer:
15          M_raw ← ZERO_MAP(H, W)
16          FOR EACH pixel p DO
17              IF I_max(p) > τ_intensity AND S(p) < τ_saturation THEN
18                  M_raw(p) ← I_max(p) × (1 - S(p))
19              END IF
20          END FOR
21
22      CASE Hybrid:
23          S_tan ← SOFT_STEP(I_min, τ_min)
24                  × SOFT_STEP(I_max, τ_bright)
25          S_shafer ← SOFT_STEP(I_max, τ_intensity)
26                     × SOFT_STEP(τ_saturation - S, 0)
27          S_gray  ← NORMALIZE(I_max × (1 - S))
28          I_local ← BOX_BLUR(I_max, radius = 3)
29          S_local ← NORMALIZE(MAX(I_max - I_local, 0))
30          M_raw ← 0.35 S_tan + 0.35 S_shafer
31                  + 0.20 S_gray + 0.10 S_local
32  END SWITCH
33
34  M_score ← NORMALIZE_TO_UNIT_INTERVAL(M_raw)
35  RETURN M_score
```

## Algorithm 3: Xây dựng Reflection Confidence Map

**Input**

- Reflection Score map \(M_{\mathrm{score}}\).
- Bán kính làm mượt \(r\).
- Phân vị confidence \(q\).
- Số mũ \(\gamma\).
- Ngưỡng confidence tối thiểu \(\tau_{\mathrm{conf}}\) nếu áp dụng trong training.

**Output**

- Confidence map bảo thủ \(M_{\mathrm{conf}}\in[0,1]^{H\times W}\).

```text
01  M_conf ← REPLACE_NAN_AND_INFINITY(M_score, value = 0)
02  M_conf ← CLAMP(M_conf, lower = 0)
03  M_conf ← NORMALIZE_TO_UNIT_INTERVAL(M_conf)
04
05  IF r > 0 THEN
06      M_conf ← BOX_BLUR(M_conf, radius = r)
07      M_conf ← NORMALIZE_TO_UNIT_INTERVAL(M_conf)
08  END IF
09
10  IF 0 < q < 1 AND MAX(M_conf) > 0 THEN
11      pivot ← QUANTILE(M_conf, q)
12      IF MAX(M_conf) > pivot + ε THEN
13          M_conf ← CLAMP(
14              (M_conf - pivot) / (MAX(M_conf) - pivot + ε), 0, 1
15          )
16      ELSE
17          M_conf ← INDICATOR(M_conf ≥ pivot)
18      END IF
19  END IF
20
21  IF γ > 0 AND γ ≠ 1 THEN
22      M_conf ← M_conf^γ
23  END IF
24
25  IF τ_conf > 0 THEN
26      M_conf(p) ← 0 FOR ALL p SUCH THAT M_conf(p) < τ_conf
27  END IF
28
29  M_conf ← NORMALIZE_TO_UNIT_INTERVAL(M_conf)
30  RETURN M_conf
```

> **Ghi chú:** Extraction script lưu cả `ref_score` và `ref_conf` theo từng camera. Nó không tự chiếu hai map này lên không gian 3D.

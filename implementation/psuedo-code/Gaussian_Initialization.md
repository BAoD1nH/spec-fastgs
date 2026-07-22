# Khởi tạo Gaussian và ASG

## Algorithm 4: Khởi tạo biểu diễn Spec-FastGS

**Input**

- Point cloud \(P=\{(\mathbf{x}_i,\mathbf{c}_i)\}_{i=1}^{N}\).
- Bậc SH tối đa \(D_{\mathrm{SH}}\).
- Số chiều ASG feature \(D_{\mathrm{ASG}}\).

**Output**

- Mô hình Gaussian \(G\).
- Mạng specular dùng chung \(F_{\mathrm{spec}}\).

```text
01  FOR EACH point p_i = (x_i, c_i) ∈ P DO
02      μ_i         ← x_i
03      f_DC,i      ← RGB_TO_SH(c_i)
04      f_SH-rest,i ← ZERO_VECTOR(SH_REST_DIMENSION(D_SH))
05      f_ASG,i     ← ZERO_VECTOR(D_ASG)
06
07      d_i ← NEAREST_NEIGHBOR_DISTANCE(P, p_i)
08      s_i ← LOG(SQRT(MAX(d_i², ε))) × [1, 1, 1]
09      r_i ← [1, 0, 0, 0]                    // identity quaternion
10      α_i ← 0.1
11  END FOR
12
13  G ← {
14      positions       = {μ_i},
15      scales          = {s_i},
16      rotations       = {r_i},
17      opacities       = {α_i},
18      SH features     = {f_DC,i, f_SH-rest,i},
19      ASG features    = {f_ASG,i}
20  }
21
22  F_spec ← INITIALIZE_SHARED_SPECULAR_MLP(D_ASG)
23  RETURN G, F_spec
```

> **Ghi chú:** \(N_{\mathrm{Gaussian,init}}=|P|\). Reflection Score không thêm Gaussian và không gán nhãn specular trong initialization hiện tại.

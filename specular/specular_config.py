# fastgs/specular/specular_config.py

# ---- View sampling ----
NUM_SAMPLE_VIEWS = 8        # đủ để thấy structure theo view

# ---- Pixel-level thresholds (L1, normalized [0,1]) ----
PIXEL_ERROR_THRESH = 0.06   # coi là "high error" tại pixel

# ---- Gaussian-level variance control ----
MIN_VIEWS_FOR_STATS = 1     # cần ít nhất 1 view để kết luận
MIN_PEAK_VIEWS = 1          # specular có thể chỉ peak ở 1-2 view

VARIANCE_THRESH = 1e-7    # threshold variance theo view
PEAK_RATIO_THRESH = 1.2     # max(error) / mean(error)

# ---- Sparsity control ----
MAX_SPECULAR_RATIO = 0.1

# ---- Debug ----
PRINT_STATS = True
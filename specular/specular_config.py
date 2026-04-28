# specular/specular_config.py

# ---- Pixel-level spotlight detection ----
PIXEL_TOPK_RATIO = 0.001      # top 0.1% brightest error pixels
PIXEL_ERROR_THRESH = 0.05     # optional absolute threshold

# ---- Pixel -> Gaussian assignment ----
MIN_PIXEL_PER_GAUSSIAN = 5    # Gaussian must explain ≥5 hotspot pixels

# ---- Sparsity control ----
MAX_SPECULAR_RATIO = 0.03     # ≤3% Gaussians

# ---- View sampling ----
NUM_SAMPLE_VIEWS = 8

# ---- Debug ----
PRINT_STATS = True
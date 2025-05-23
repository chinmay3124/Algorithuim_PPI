import numpy as np
from numpy import sqrt

# ─────────────────────────────────────────────────────────────────────────────
# Method B: Map DepthAnything scores → Real‐world distance via your PPI eqn
# ─────────────────────────────────────────────────────────────────────────────

# 1) Calibration data: distances (in inches) and corresponding depth scores
#    Replace the depth_scores array with your observed mean depth values.
distances    = np.array([10, 12, 14, 16, 18, 20], dtype=float)
depth_scores = np.array([44.870, 39.652, 38.986, 35.891, 34.597, 31.644], dtype=float)
# e.g. depth_scores = np.array([42.1,  Forty,  ...])  # fill in your actual numbers

# 2) Your existing PPI vs. distance polynomial:
#    PPI(d) = a·d² + b·d + c
#    Replace a, b, c with your discovered coefficients.
a, b, c = 0.6937, -29.3668, 388.2143  # ← example values; use your real ones

# 3) Compute the PPI that your model predicts at each calibration distance
ppi_vals = a*distances**2 + b*distances + c

# 4) Check correlation between depth score and PPI
corr_matrix = np.corrcoef(depth_scores, ppi_vals)
r_value     = corr_matrix[0,1]
print(f"Pearson r between depth_score and theoretical PPI: {r_value:.3f}")

# 5) Fit a linear mapping: PPI ≈ m·(depth_score) + b0
m, b0 = np.polyfit(depth_scores, ppi_vals, deg=1)
print(f"Linear fit: PPI ≈ {m:.4f}·score + {b0:.4f}")

# 6) Define a function to estimate distance from a new depth score:
def estimate_distance_from_score(score):
    # 6a) predict PPI from the depth score
    ppi_pred = m*score + b0

    # 6b) solve the quadratic a·d² + b·d + (c - ppi_pred) = 0
    A = a
    B = b
    C = c - ppi_pred

    disc = B*B - 4*A*C
    if disc < 0:
        raise ValueError("Negative discriminant: no real solution for distance")

    root1 = (-B + sqrt(disc)) / (2*A)
    root2 = (-B - sqrt(disc)) / (2*A)

    # choose the root that lies in your calibration range
    for d in (root1, root2):
        if distances.min() - 1e-3 <= d <= distances.max() + 1e-3:
            return d
    # if neither root is in range, return both
    return root1, root2

# 7) Example usage with a new depth score
new_score = 60.5  # replace with your actual new model score
est_d = estimate_distance_from_score(new_score)
print(f"Estimated distance for score {new_score:.3f}: {est_d} inches")

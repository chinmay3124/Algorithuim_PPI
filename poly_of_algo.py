import numpy as np

# 1) Your measured data :
distances = np.array([10, 12, 14, 16,  18, 20])
ppis      = np.array([166, 132.4, 113.4, 97.3, 85.3, 77.4])

# 2) Fit degree-2 poly: solves for [a, b, c]
coefs = np.polyfit(distances, ppis, deg=2)  
poly  = np.poly1d(coefs)

# 3) Print the equation
a, b, c = coefs
print(f"PPI(d) = {a:.4f}·d² + {b:.4f}·d + {c:.4f}")

# 4) Predict at a new distance
d_new = 17
print(f"Predicted PPI at {d_new}in: {poly(d_new):.1f}")
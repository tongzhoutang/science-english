# Unit P1 Code Lab: fitting Taylor's blast-wave data
import numpy as np
import matplotlib.pyplot as plt

# Taylor (1950), Table 1: time in ms, radius in m
t_ms = np.array([0.10, 0.24, 0.38, 0.52, 0.66, 0.80, 0.94, 1.08, 1.22,
                 1.36, 1.50, 1.65, 1.79, 1.93, 3.26, 3.53, 3.80, 4.07,
                 4.34, 4.61, 15.0, 25.0, 34.0, 53.0, 62.0])
R = np.array([11.1, 19.9, 25.4, 28.8, 31.9, 34.2, 36.3, 38.9, 41.0,
              42.8, 44.4, 46.0, 46.9, 48.7, 59.0, 61.1, 62.9, 64.3,
              65.6, 67.3, 106.5, 130.0, 145.0, 175.0, 185.0])
t = t_ms / 1000            # convert ms to s

# ---- Things to change ----
t_min = 0.2                # ignore points earlier than this (ms)
K = 0.856                  # K = 0.856 for gamma = 1.4; K = 1.167 for gamma = 1.3
rho = 1.25                 # density of air, kg/m^3
# --------------------------

keep = t_ms >= t_min
log_t, log_R = np.log10(t[keep]), np.log10(R[keep])

# 1. Let the data choose the slope
slope, intercept = np.polyfit(log_t, log_R, 1)
print(f"points used: {keep.sum()}")
print(f"free fit: slope = {slope:.3f}")

# 2. Fix the slope at 2/5 and find Q = R^5 / t^2
log_c = np.mean(log_R - 0.4 * log_t)
Q = 10 ** (5 * log_c)
E = K * rho * Q
print(f"Q = {Q:.2e} m^5/s^2")
print(f"E = {E:.2e} J = {E / 4.184e12:.1f} kt of TNT")

# 3. Plot the data and both lines
plt.loglog(t_ms, R, "k+", label="Taylor (1950)")
line_t = np.logspace(-4, -1, 50)
plt.loglog(line_t * 1000, 10**log_c * line_t**0.4, label="slope 2/5")
plt.loglog(line_t * 1000, 10**intercept * line_t**slope, "--", label="free fit")
plt.xlabel("t (ms)")
plt.ylabel("R (m)")
plt.legend()
plt.savefig("taylor_fit.png", dpi=150)
plt.show()

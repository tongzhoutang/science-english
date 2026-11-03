# P1 figure data and number checks: Sedov-Taylor self-similar solution and Taylor (1950) Table 1.
# Run:  python figures/P1/make_data.py   (from science-english/)
import os
import numpy as np
from scipy.integrate import solve_ivp

here = os.path.dirname(os.path.abspath(__file__))


def sedov(gamma, n=901):
    """Integrate inward from the shock. rho = rho0 G(x), v = U F(x), p = rho0 U^2 P(x), x = r/R."""
    # U = dR/dt, dU/dt = -(3/2) U^2 / R. With g = G'/G and h = P'/P, w = F - x:
    #   mass:     F' = -w g - 2F/x
    #   entropy:  h = gamma g + 3/w
    #   momentum: w F' - 1.5 F + (P/G) h = 0  (linear in g)
    def rhs(x, y):
        F, lnG, lnP = y
        w = F - x
        ratio = np.exp(lnP - lnG)
        g = (2 * F * w / x + 1.5 * F - ratio * 3 / w) / (ratio * gamma - w * w)
        return [-w * g - 2 * F / x, g, gamma * g + 3 / w]

    y0 = [2 / (gamma + 1), np.log((gamma + 1) / (gamma - 1)), np.log(2 / (gamma + 1))]
    xs = np.linspace(1.0, 0.1, n)
    sol = solve_ivp(rhs, (1.0, 0.1), y0, t_eval=xs, rtol=1e-9, atol=1e-11, method="Radau")
    assert sol.success, sol.message
    x, F, G, P = sol.t, sol.y[0], np.exp(sol.y[1]), np.exp(sol.y[2])
    # E = K rho0 R^5 / t^2, K = 4 pi (2/5)^2 int (G F^2/2 + P/(gamma-1)) x^2 dx
    kin = -np.trapezoid(G * F**2 / 2 * x**2, x)
    thermal = -np.trapezoid(P / (gamma - 1) * x**2, x) + P[-1] / (gamma - 1) * x[-1] ** 3 / 3
    K = 4 * np.pi * 0.16 * (kin + thermal)
    mass = -3 * np.trapezoid(G * x**2, x)
    return dict(x=x, F=F, G=G, P=P, K=K, S=K ** (-0.2), kin_frac=kin / (kin + thermal), mass=mass)


for g in (1.2, 1.3, 1.4, 5 / 3):
    s = sedov(g)
    print(f"gamma={g:.3f} K={s['K']:.4f} S={s['S']:.4f} kin_frac={s['kin_frac']:.3f} mass_check={s['mass']:.4f}")

s = sedov(1.4)
x, G, P, F = s["x"], s["G"], s["P"], s["F"]
sel = np.arange(0, len(x), 10)
np.savetxt(os.path.join(here, "sedov_profile.dat"),
           np.c_[x[sel], G[sel] / G[0], P[sel] / P[0], F[sel] / F[0]],
           header="x rho p v", comments="", fmt="%.5f")
for x0 in (0.95, 0.9, 0.8):
    m = x >= x0
    print(f"mass fraction in x>={x0}: {-3 * np.trapezoid((G * x**2)[m], x[m]):.3f}")
print("central p/p2 =", P[-1] / P[0])

# Taylor (1950) Part II, Table 1
t_ms = np.array([0.10, 0.24, 0.38, 0.52, 0.66, 0.80, 0.94, 1.08, 1.22, 1.36, 1.50, 1.65, 1.79, 1.93,
                 3.26, 3.53, 3.80, 4.07, 4.34, 4.61, 15.0, 25.0, 34.0, 53.0, 62.0])
R = np.array([11.1, 19.9, 25.4, 28.8, 31.9, 34.2, 36.3, 38.9, 41.0, 42.8, 44.4, 46.0, 46.9, 48.7,
              59.0, 61.1, 62.9, 64.3, 65.6, 67.3, 106.5, 130.0, 145.0, 175.0, 185.0])
t = t_ms * 1e-3
np.savetxt(os.path.join(here, "taylor_table.dat"), np.c_[t_ms, R], header="t R", comments="", fmt="%.2f %.1f")

for name, m in (("all", np.ones_like(t, bool)), ("drop first", t_ms > 0.1)):
    b, a = np.polyfit(np.log10(t[m]), np.log10(R[m]), 1)
    Q = 10 ** (2.5 * np.mean(np.log10(R[m]) - 0.4 * np.log10(t[m])))
    print(f"{name}: n={m.sum()} slope={b:.4f} intercept={a:.4f} Q(free fit)={10**(5*a):.3e} Q(slope 2/5)={Q:.3e}")

rho, TNT, c, p0 = 1.25, 4.2e12, 340.0, 1.013e5
print("Taylor: 6.67e13 * K=0.856 * rho ->", 0.856 * rho * 6.67e13, "J =", 0.856 * rho * 6.67e13 / TNT, "kt")
for i in (0, 1, 6, 13, 20, 24):
    q = R[i] ** 5 / t[i] ** 2
    U = 0.4 * R[i] / t[i]
    p2 = 2 * rho * U**2 / 2.4
    print(f"t={t_ms[i]} R={R[i]} Q={q:.3e} kt(K=.856)={0.856 * rho * q / TNT:.1f} U={U:.0f} M={U / c:.1f} p2/p0={p2 / p0:.0f}"
          f" T2(ideal)={p2 / (6 * rho * 287):.0f}")

E = 7.1e13
print("R when swept mass = 5 t:", (3 * 5000 / (4 * np.pi * rho)) ** (1 / 3))
print("R* = (E/p0)^(1/3):", (E / p0) ** (1 / 3), "t* =", E ** (1 / 3) * rho**0.5 * p0 ** (-5 / 6))
S = s["S"]
ts = np.logspace(-4, 1, 20001)
Rs = S * (E / rho) ** 0.2 * ts**0.4
Us = 0.4 * Rs / ts
for M in (3, 2):
    i = np.argmin(abs(Us - M * c))
    print(f"M={M} at t={ts[i]:.3f} s, R={Rs[i]:.0f} m")

# Supernova remnant
msun, yr, pc, mp, k = 1.989e30, 3.156e7, 3.086e16, 1.673e-27, 1.381e-23
E, rho_ism, tt = 1e44, 1e6 * mp, 1000 * yr
S53 = sedov(5 / 3)["S"]
Rs = S53 * (E * tt**2 / rho_ism) ** 0.2
U = 0.4 * Rs / tt
print(f"SNR S={S53:.3f} R={Rs / pc:.2f} pc U={U / 1e3:.0f} km/s T={3 * 0.6 * mp * U**2 / (16 * k):.2e} K")
M = 1.4 * msun
R1 = (3 * M / (4 * np.pi * rho_ism)) ** (1 / 3)
v = (2 * E / M) ** 0.5
print(f"swept radius={R1 / pc:.2f} pc v={v / 1e3:.0f} km/s t={R1 / v / yr:.0f} yr")

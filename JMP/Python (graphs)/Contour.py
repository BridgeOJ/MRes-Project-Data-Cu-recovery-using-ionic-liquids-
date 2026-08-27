import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import t as t_dist
import pandas as pd

# ── Apply Style Guide ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "Arial",
    "font.size":            12,
    "axes.linewidth":       1.2,
    "axes.grid":            False,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.major.size":     5,
    "ytick.major.size":     5,
    "xtick.minor.size":     3,
    "ytick.minor.size":     3,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_excel('DSD_Cu_leaching_data2.xlsx')

# ── Model coefficients ────────────────────────────────────────────────────────
beta = np.array([78.0180427,  0.68082098, -1.5010575,
                 -0.0585321, -0.3555947,   0.0060675])

# ── Variance-covariance matrix ────────────────────────────────────────────────
X = np.column_stack([
    np.ones(len(df)),
    df['LS_ratio'],
    df['Time_hr'],
    df['Temp'],
    df['H2O2_pct'],
    (df['Temp'] - 57.5) ** 2
])

rmse     = 2.097323
df_resid = 6
V        = (rmse ** 2) * inv(X.T @ X)
t_crit   = t_dist.ppf(0.975, df_resid)

# ── Fixed factors at midpoints ────────────────────────────────────────────────
time_fixed = 7.0
h2o2_fixed = 17.5

# ── Grid ──────────────────────────────────────────────────────────────────────
n          = 500
ls_range   = np.linspace(10, 30, n)
temp_range = np.linspace(25, 90, n)
LS, TEMP   = np.meshgrid(ls_range, temp_range)

# ── Predicted yield at every grid point ───────────────────────────────────────
Z = (beta[0]
     + beta[1] * LS
     + beta[2] * time_fixed
     + beta[3] * TEMP
     + beta[4] * h2o2_fixed
     + beta[5] * (TEMP - 57.5) ** 2)

# ── Contour value — matches JMP default of 77.825 ────────────────────────────
contour_level = 77.825404

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

# Single contour line at the JMP contour value
cs = ax.contour(LS, TEMP, Z,
                levels=[contour_level],
                colors=['#fe201c'],      # red — matches JMP
                linewidths=1.2)

# ── Contour label ─────────────────────────────────────────────────────────────
ax.clabel(cs,
          fmt=f'{contour_level:.1f}',
          fontsize=9,
          inline=True)

# ── Crosshairs — midpoint reference lines like JMP ───────────────────────────
ax.axhline(y=57.5, color='black', linewidth=0.8,
           linestyle='-', zorder=2)
ax.axvline(x=20,   color='black', linewidth=0.8,
           linestyle='-', zorder=2)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel('LS Ratio',
              fontsize=14, fontweight='bold')
ax.set_ylabel('Temperature / °C',
              fontsize=14, fontweight='bold')
ax.set_title('Contour Profiler — Cu Yield (%)',
             fontsize=13, fontweight='bold')

ax.set_xlim(10, 30)
ax.set_ylim(25, 90)
ax.set_xticks([10, 15, 20, 25, 30])
ax.set_yticks([25, 40, 55, 70, 85])

# ── Spines and ticks ──────────────────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

ax.tick_params(which='both', top=True, right=True,
               direction='in')

plt.tight_layout()
plt.savefig('contour_profiler.png', dpi=300, bbox_inches='tight')
plt.show()

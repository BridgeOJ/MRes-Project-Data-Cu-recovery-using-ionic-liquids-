import numpy as np
import matplotlib.pyplot as plt

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

# ── Data ──────────────────────────────────────────────────────────────────────
# Design runs only (R1-R8) — R9 moved to centre points
actual_runs    = [81.39, 72.86, 77.73, 80.04, 66.95,
                  74.45, 65.41, 90.24]
predicted_runs = [82.21, 73.68, 76.79, 79.11, 67.79,
                  75.29, 65.54, 90.36]

# Centre points — R9 + CP1-CP3 (all at identical centre condition)
actual_cp    = [69.18, 75.48, 72.19, 70.97]
predicted_cp = [71.54, 71.54, 71.54, 71.54]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

# ── Design runs — filled blue circles ────────────────────────────────────────
ax.scatter(predicted_runs, actual_runs,
           color='#2724fd',
           s=55,
           marker='o',
           edgecolors='black',
           linewidths=0.6,
           zorder=4,
           label='Design runs (R1–R8)')

# ── Centre points — red diamonds (R9 + CP1-CP3) ───────────────────────────
ax.scatter(predicted_cp, actual_cp,
           color='#fe201c',
           s=70,
           marker='D',
           edgecolors='black',
           linewidths=0.6,
           zorder=4,
           label='Centre points (R9, CP1–CP3)')

# ── y = x reference line ──────────────────────────────────────────────────────
lims = [60, 95]
ax.plot(lims, lims,
        color='black',
        linewidth=1.2,
        linestyle='--',
        zorder=2)

# ── y = x inline label ────────────────────────────────────────────────────────

lims = [60, 95]
ax.plot(lims, lims,
        color='black',
        linewidth=1.2,
        linestyle='--',
        zorder=2,
        label='Perfect fit (y = x)')


# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel('Predicted Cu Yield / %',
              fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Cu Yield / %',
              fontsize=14, fontweight='bold')
#ax.set_title('Actual vs Predicted — Cu Yield (%)',
     #        fontsize=13, fontweight='bold')

ax.set_xlim(60, 95)
ax.set_ylim(60, 95)
ax.set_xticks(np.arange(60, 96, 5))
ax.set_yticks(np.arange(60, 96, 5))

# ── Spines and ticks ──────────────────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

ax.tick_params(which='both', top=True, right=True,
               direction='in')

# ── Legend — two entries only ─────────────────────────────────────────────────
legend = ax.legend(
    fontsize=10,
    frameon=False,
    loc='upper left',
    handletextpad=0.5,
    labelspacing=0.4,
)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

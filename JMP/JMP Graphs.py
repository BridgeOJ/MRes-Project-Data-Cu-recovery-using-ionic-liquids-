import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

# ── Style guide ───────────────────────────────────────────────────────────────
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
    "legend.frameon":       False,
    "legend.framealpha":    1.0,
    "legend.edgecolor":     "black",
    "legend.fontsize":      10,
})

# ── 1. Model & centre points ──────────────────────────────────────────────────
CP = dict(LS=20, time=7, T=57.5, H2O2=17.5)

# ── 2. OLS CI ─────────────────────────────────────────────────────────────────
runs = [
    [30,    7,   25,   30],
    [10,    7,   90,    5],
    [20,   12,   25,    5],
    [20,    2,   90,   30],
    [10,    2,  57.5,  30],
    [30,   12,  57.5,   5],
    [10,   12,   25,  17.5],
    [30,    2,   90,  17.5],
    [20,    7,  57.5, 17.5],
]

y_obs = np.array([81.39523493, 72.86409423, 77.72893703, 80.04505419,
                  66.954955,   74.4506681,  65.41420124, 90.23660617,
                  69.17718828])

def build_X(data):
    data = np.array(data, dtype=float)
    LS, time, T, H2O2 = data[:,0], data[:,1], data[:,2], data[:,3]
    return np.column_stack([np.ones(len(data)), LS, time, T, (T-57.5)**2, H2O2])

X_mat   = build_X(runs)
n, p    = X_mat.shape
XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
beta    = XtX_inv @ X_mat.T @ y_obs
resid   = y_obs - X_mat @ beta
s2      = resid @ resid / (n - p)
s       = np.sqrt(s2)
t_crit  = t_dist.ppf(0.975, df=n - p)

def ci_band(rows):
    X_new = np.array(rows, dtype=float)
    pred  = X_new @ beta
    lev   = np.einsum('ij,jk,ik->i', X_new, XtX_inv, X_new)
    half  = t_crit * s * np.sqrt(lev)
    return pred, pred - half, pred + half

# ── Centre point predicted yield from OLS beta ────────────────────────────────
cp_row   = np.array([[1, CP['LS'], CP['time'], CP['T'],
                       (CP['T']-57.5)**2, CP['H2O2']]])
cp_yield = float((cp_row @ beta)[0])
print(f"OLS centre point Cu yield: {cp_yield:.4f} %")

# ── 3. Factor definitions — 2x2 layout ───────────────────────────────────────
# row, col indices for 2x2 grid
factors = [
    dict(label='L/S Ratio',        key='LS',   cp=20,   lo=10, hi=30,
         ticks=[10,15,20,25,30],   panel='(a)', row=0, col=0),
    dict(label='Time (h)',         key='time', cp=7,    lo=2,  hi=12,
         ticks=[2,4,6,8,10,12],    panel='(b)', row=0, col=1),
    dict(label='Temperature (°C)',            key='T',    cp=57.5, lo=25, hi=90,
         ticks=[25,40,60,80],      panel='(c)', row=1, col=0),
    dict(label='H$_2$O$_2$ / % v/v', key='H2O2', cp=17.5, lo=5,  hi=30,
         ticks=[5,10,15,20,25,30], panel='(d)', row=1, col=1),
]

# ── 4. Fixed y limits ─────────────────────────────────────────────────────────
y_min = 60
y_max = 95

# ── 5. Figure — 2x2 grid ──────────────────────────────────────────────────────
RED = '#fe201c'

fig, axes = plt.subplots(2, 2,
                         figsize=(9, 9),
                         gridspec_kw=dict(wspace=0.08, hspace=0.20))
fig.patch.set_facecolor('white')

for fac in factors:
    ax  = axes[fac['row'], fac['col']]
    col = fac['col']
    row = fac['row']

    xs = np.linspace(fac['lo'], fac['hi'], 300)
    rows = []
    for x in xs:
        v = dict(LS=CP['LS'], time=CP['time'], T=CP['T'], H2O2=CP['H2O2'])
        v[fac['key']] = x
        rows.append([1, v['LS'], v['time'], v['T'], (v['T']-57.5)**2, v['H2O2']])

    pred, lo95, hi95 = ci_band(rows)

    # ── Set limits first ──
    ax.set_xlim(fac['lo'], fac['hi'])
    ax.set_ylim(y_min, y_max)

    # ── Reference lines first so they sit behind data ──
    ax.vlines(fac['cp'], y_min, y_max,
              colors=RED, linewidth=1.4, linestyle='--', zorder=2)
    ax.hlines(cp_yield, fac['lo'], fac['hi'],
              colors=RED, linewidth=1.4, linestyle='--', zorder=2)

    # ── CI band + border lines ──
    ax.fill_between(xs, lo95, hi95, color='steelblue', alpha=0.25,
                    linewidth=0, zorder=3)
    ax.plot(xs, lo95, color='steelblue', linewidth=1.4, zorder=3)
    ax.plot(xs, hi95, color='steelblue', linewidth=1.4, zorder=3)

    # ── Prediction curve ──
    ax.plot(xs, pred, color='black', linewidth=2.2, zorder=4)

    # ── Square panels ──
    ax.set_box_aspect(1)

    # ── Spines & ticks ──
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    ax.tick_params(which='both', top=True, right=True,
                   left=True, bottom=True, direction='in',
                   labelsize=13)

    # ── X ticks & label — all panels ──
    ax.set_xticks(fac['ticks'])
    ax.set_xlabel(fac['label'], fontsize=14, fontweight='bold', labelpad=8)

    # ── Y axis — left column only ──
    ax.set_yticks([65, 75, 85, 95])
    if col == 0:
        ax.set_ylabel('Cu yield / %', fontsize=14, fontweight='bold')
        ax.tick_params(labelleft=True)
    else:
        ax.tick_params(labelleft=False)

    # ── Panel label ──
    ax.text(0.04, 0.96, fac['panel'],
            transform=ax.transAxes,
            fontsize=20, fontweight='bold',
            va='top', ha='left', color='black')

plt.savefig('cu_yield_profiler.png', dpi=150, bbox_inches='tight')
plt.show()

"""
Generate high-quality, publication-ready figures for the ADAA IEEE paper.
Run from project root: python docs/generate_figures.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

BERKELEY = os.path.join(os.path.dirname(__file__), '..', 'data', 'berkeley_admissions.csv')
CHURN    = os.path.join(os.path.dirname(__file__), '..', 'data', 'simpsons_churn.csv')
ADULT    = os.path.join(os.path.dirname(__file__), '..', 'data', 'adult_census.csv')

# Clean, high-contrast color palette
C = {
    'male':        '#1565C0',
    'female':      '#AD1457',
    'global':      '#E64A19',
    'startup':     '#6A1B9A',
    'midmarket':   '#00838F',
    'enterprise':  '#2E7D32',
    'robust':      '#2E7D32',
    'confounded':  '#C62828',
    'paradox':     '#F57F17',
    'bg':          '#FAFAFA',
    'grid':        '#E0E0E0',
    'text':        '#212121',
    'subtext':     '#616161',
    'box_blue':    '#1565C0',
    'box_red':     '#B71C1C',
    'box_green':   '#1B5E20',
    'box_gray':    '#455A64',
    'box_dark':    '#37474F',
}

# Global style — clean sans-serif, no top/right spines
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          13,
    'axes.titlesize':     14,
    'axes.titleweight':   'bold',
    'axes.labelsize':     12,
    'axes.labelcolor':    C['text'],
    'axes.edgecolor':     '#BDBDBD',
    'axes.facecolor':     C['bg'],
    'axes.grid':          True,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'grid.color':         C['grid'],
    'grid.linewidth':     0.8,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'xtick.color':        C['subtext'],
    'ytick.color':        C['subtext'],
    'figure.facecolor':   'white',
    'figure.dpi':         200,
    'legend.framealpha':  0.9,
    'legend.fontsize':    10,
    'legend.edgecolor':   '#BDBDBD',
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    kb = os.path.getsize(path) // 1024
    print(f"  -> {name}  ({kb} KB)")

# =============================================================================
# FIG 1  UC Berkeley: Admission rates by department + paradox illustration
# =============================================================================
print("\n[1/5] Berkeley admission rates...")
berk  = pd.read_csv(BERKELEY)
depts = sorted(berk['department'].unique())

male_rates, female_rates, ns = [], [], []
for d in depts:
    sub = berk[berk['department'] == d]
    male_rates.append(sub[sub['gender'] == 'Male']['admitted'].mean() * 100)
    female_rates.append(sub[sub['gender'] == 'Female']['admitted'].mean() * 100)
    ns.append(len(sub))

global_male   = berk[berk['gender'] == 'Male']['admitted'].mean() * 100
global_female = berk[berk['gender'] == 'Female']['admitted'].mean() * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# --- Panel A: grouped bars ---
x = np.arange(len(depts))
w = 0.35
b1 = ax1.bar(x - w/2, male_rates,   w, label='Male',   color=C['male'],   alpha=0.90, zorder=3, linewidth=0)
b2 = ax1.bar(x + w/2, female_rates, w, label='Female', color=C['female'], alpha=0.90, zorder=3, linewidth=0)

# Global dashed reference lines
ax1.axhline(global_male,   color=C['male'],   linestyle='--', linewidth=1.6, alpha=0.7, zorder=2)
ax1.axhline(global_female, color=C['female'], linestyle='--', linewidth=1.6, alpha=0.7, zorder=2)
ax1.text(5.6, global_male + 1.5,   f'Global M: {global_male:.1f}%',   color=C['male'],   fontsize=9.5, va='bottom')
ax1.text(5.6, global_female - 4.0, f'Global F: {global_female:.1f}%', color=C['female'], fontsize=9.5, va='top')

# Reversal annotations
for i, (m, f) in enumerate(zip(male_rates, female_rates)):
    if f > m:
        ax1.text(x[i] + w/2, f + 2, 'F > M', ha='center', fontsize=8.5,
                 color=C['female'], fontweight='bold')
    ax1.text(x[i] - w/2, m + 1,   f'{m:.0f}%',  ha='center', fontsize=8, color='white' if m > 15 else C['text'])
    ax1.text(x[i] + w/2, f + 1,   f'{f:.0f}%',  ha='center', fontsize=8, color='white' if f > 15 else C['text'])

ax1.set_xticks(x)
ax1.set_xticklabels([f'Dept {d}\n(N={n})' for d, n in zip(depts, ns)], fontsize=10)
ax1.set_ylabel('Admission Rate (%)', fontsize=12)
ax1.set_ylim(0, 102)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_title('(a) Department-Level Admission Rates by Gender', pad=10)
ax1.legend(loc='upper right')
ax1.set_axisbelow(True)

# --- Panel B: Aggregate vs Stratified ---
weighted_f = sum(
    berk[(berk['department'] == d) & (berk['gender'] == 'Female')]['admitted'].mean() *
    len(berk[(berk['department'] == d) & (berk['gender'] == 'Female')])
    for d in depts
) / len(berk[berk['gender'] == 'Female']) * 100

cats   = ['Aggregate View\n(Explorer Output)', 'Stratified View\n(Devil\'s Advocate)']
m_vals = [global_male, global_male]
f_vals = [global_female, weighted_f]

x2 = np.array([0, 1])
ax2.bar(x2 - 0.2, m_vals, 0.35, color=C['male'],   alpha=0.9, label='Male',   zorder=3, linewidth=0)
ax2.bar(x2 + 0.2, f_vals, 0.35, color=C['female'], alpha=0.9, label='Female', zorder=3, linewidth=0)

for xi, mv, fv in zip(x2, m_vals, f_vals):
    ax2.text(xi - 0.2, mv + 0.8, f'{mv:.1f}%', ha='center', fontsize=10, color=C['male'],   fontweight='bold')
    ax2.text(xi + 0.2, fv + 0.8, f'{fv:.1f}%', ha='center', fontsize=10, color=C['female'], fontweight='bold')

# Arrow showing paradox resolution
ax2.annotate('', xy=(1.2, weighted_f), xytext=(1.2, global_female),
             arrowprops=dict(arrowstyle='->', color=C['paradox'], lw=2.5))
mid = (global_female + weighted_f) / 2
ax2.text(1.28, mid, 'Paradox\nrevealed', fontsize=10, color=C['paradox'],
         fontweight='bold', va='center')

ax2.set_xticks([0, 1])
ax2.set_xticklabels(cats, fontsize=11)
ax2.set_ylabel('Admission Rate (%)', fontsize=12)
ax2.set_ylim(0, 55)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_title("(b) Aggregate vs. Stratified: Simpson's Paradox Exposed", pad=10)
ax2.legend(loc='upper right')
ax2.set_axisbelow(True)

fig.suptitle("Fig. 1  —  UC Berkeley Graduate Admissions (1973): Visualizing Simpson's Paradox",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=3)
save(fig, 'fig1_berkeley_paradox.png')


# =============================================================================
# FIG 2  Churn: scatter with per-tier regressions + correlation sign reversal
# =============================================================================
print("[2/5] Churn paradox...")
churn = pd.read_csv(CHURN)
tiers       = ['Startup', 'Mid-Market', 'Enterprise']
tier_colors = [C['startup'], C['midmarket'], C['enterprise']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel A: Scatter + regression ---
for tier, col in zip(tiers, tier_colors):
    sub = churn[churn['tier'] == tier]
    ax1.scatter(sub['support_calls'], sub['churn'],
                alpha=0.18, s=18, color=col, zorder=2)
    m, b = np.polyfit(sub['support_calls'], sub['churn'], 1)
    xs = np.linspace(sub['support_calls'].min(), sub['support_calls'].max(), 200)
    ax1.plot(xs, m * xs + b, color=col, linewidth=2.8, label=f'{tier} (r={np.corrcoef(sub["support_calls"], sub["churn"])[0,1]:+.3f})', zorder=3)

# Global trend
m_g, b_g = np.polyfit(churn['support_calls'], churn['churn'], 1)
xs_g = np.linspace(churn['support_calls'].min(), churn['support_calls'].max(), 200)
r_g  = np.corrcoef(churn['support_calls'], churn['churn'])[0, 1]
ax1.plot(xs_g, m_g * xs_g + b_g, color=C['global'], linewidth=3.0,
         linestyle='--', label=f'Global aggregate (r={r_g:+.3f})', zorder=4)

ax1.set_xlabel('Support Calls', fontsize=12)
ax1.set_ylabel('Churn  (0 = Stay,  1 = Leave)', fontsize=12)
ax1.set_title('(a) Support Calls vs. Churn\nGlobal Trend vs. Within-Tier Trends', pad=10)
ax1.legend(loc='lower left', fontsize=10)
ax1.set_axisbelow(True)

# Annotation box
ax1.text(0.98, 0.97,
         'Global trend (--) says:\nsupport reduces churn\n\nWithin every tier:\nsupport INCREASES churn',
         transform=ax1.transAxes, ha='right', va='top', fontsize=9.5,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor=C['paradox'], linewidth=1.5))

# --- Panel B: Correlation bar chart ---
tier_corrs   = []
for tier in tiers:
    sub = churn[churn['tier'] == tier]
    tier_corrs.append(np.corrcoef(sub['support_calls'], sub['churn'])[0, 1])

labels    = ['Global\n(Aggregate)'] + tiers
corrs     = [r_g] + tier_corrs
bar_cols  = [C['confounded'] if c < 0 else C['robust'] for c in corrs]
bar_alpha = [1.0, 0.85, 0.85, 0.85]

bars = ax2.bar(labels, corrs, color=bar_cols, alpha=0.9, width=0.5,
               zorder=3, linewidth=0, edgecolor='white')
ax2.axhline(0, color=C['text'], linewidth=1.0, zorder=4)

for bar, v in zip(bars, corrs):
    ypos = v + 0.012 if v >= 0 else v - 0.015
    va   = 'bottom' if v >= 0 else 'top'
    ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
             f'r = {v:+.3f}', ha='center', va=va, fontsize=11, fontweight='bold',
             color=C['confounded'] if v < 0 else C['robust'])

ax2.set_ylabel("Pearson r   (support_calls vs churn)", fontsize=12)
ax2.set_title("(b) Correlation Sign Reversal\nGlobal (Negative) vs. Within-Tier (Positive)", pad=10)
ax2.set_ylim(-0.42, 0.45)
ax2.set_axisbelow(True)

neg_p = mpatches.Patch(color=C['confounded'], label='Negative (misleading aggregate)')
pos_p = mpatches.Patch(color=C['robust'],     label='Positive (true within-tier effect)')
ax2.legend(handles=[neg_p, pos_p], loc='upper right', fontsize=10)

fig.suptitle("Fig. 2  —  Commercial Churn Benchmark: Support Calls vs. Churn Simpson's Paradox",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=3)
save(fig, 'fig2_churn_paradox.png')


# =============================================================================
# FIG 3  Adult Census: effect sizes + verdict distribution
# =============================================================================
print("[3/5] Adult Census summary...")
findings = [
    ('education_num',  0.3729, 'ROBUST'),
    ('occupation',     0.820,  'ROBUST'),
    ('hours_per_week', 0.2253, 'CONFOUNDED'),
    ('race',           0.148,  'PARADOX'),
    ('gender',         0.113,  'PARADOX'),
    ('age',            0.0300, 'PARADOX'),
]
vc = {'ROBUST': C['robust'], 'CONFOUNDED': C['confounded'], 'PARADOX': C['paradox']}
vl = {'ROBUST': 'ROBUST & VERIFIED', 'CONFOUNDED': 'SPURIOUS / CONFOUNDED', 'PARADOX': "SIMPSON'S PARADOX"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel A: horizontal bars ---
vars_   = [f[0] for f in findings]
effs    = [f[1] for f in findings]
vcols   = [vc[f[2]] for f in findings]
vlabels = [vl[f[2]] for f in findings]
y_pos   = np.arange(len(vars_))

bars = ax1.barh(y_pos, effs, color=vcols, alpha=0.88, height=0.55,
                zorder=3, linewidth=0)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(vars_, fontsize=12)
ax1.set_xlabel('Effect Size  |r|  or  normalized F', fontsize=12)
ax1.set_title('(a) Explorer-Claimed Effect Sizes\nColored by Referee Verdict', pad=10)
ax1.set_axisbelow(True)

for bar, v, verdict in zip(bars, effs, [f[2] for f in findings]):
    ax1.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
             vl[verdict], va='center', fontsize=9.5, color=vc[verdict], fontweight='bold')

patches = [mpatches.Patch(color=vc[v], label=vl[v]) for v in ['ROBUST', 'CONFOUNDED', 'PARADOX']]
ax1.legend(handles=patches, loc='lower right', fontsize=10)
ax1.set_xlim(0, 1.3)

# --- Panel B: pie chart across all 9 findings ---
all_verdicts = {'ROBUST & VERIFIED': 3, "SIMPSON'S PARADOX": 5, 'SPURIOUS / CONFOUNDED': 1}
pie_colors   = [C['robust'], C['paradox'], C['confounded']]
explode      = [0.03, 0.03, 0.03]

wedges, texts, autotexts = ax2.pie(
    all_verdicts.values(),
    labels=all_verdicts.keys(),
    colors=pie_colors,
    explode=explode,
    autopct='%1.0f%%',
    startangle=120,
    wedgeprops={'edgecolor': 'white', 'linewidth': 3},
    textprops={'fontsize': 11},
    pctdistance=0.65,
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight('bold')
    at.set_color('white')

ax2.set_title('(b) Verdict Distribution\nAcross All 9 ADAA Findings (3 Benchmarks)',
              pad=12, fontsize=14, fontweight='bold')

centre = plt.Circle((0, 0), 0.40, color='white')
ax2.add_artist(centre)
ax2.text(0, 0, '9\nFindings', ha='center', va='center', fontsize=13,
         fontweight='bold', color=C['text'])

fig.suptitle("Fig. 3  —  UCI Adult Census Income Benchmark — ADAA Finding Summary",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=4)
save(fig, 'fig3_adult_census_summary.png')


# =============================================================================
# FIG 4  System Architecture
# =============================================================================
print("[4/5] System architecture...")
fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)
ax.axis('off')

def rbox(ax, x, y, w, h, title, lines, fcolor, tcolor='white', fontsize=12):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.15',
                          facecolor=fcolor, edgecolor='white',
                          linewidth=2.5, alpha=0.95, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.42, title, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=tcolor, zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.85 - i * 0.44, line,
                ha='center', va='center', fontsize=9.5,
                color=tcolor, alpha=0.90, zorder=4)

def arrow(ax, x1, y, x2, label='', color='#546E7A'):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5), zorder=5)
    if label:
        ax.text((x1 + x2) / 2, y + 0.18, label,
                ha='center', fontsize=10, color=color, style='italic', zorder=5)

# Input box
rbox(ax, 0.3, 2.6, 1.9, 1.8,
     'INPUT', ['CSV Dataset', 'Any tabular', 'data source'], C['box_gray'])

# Explorer
rbox(ax, 2.6, 1.3, 2.8, 4.2,
     'EXPLORER AGENT',
     ['Pearson r (continuous)',
      'ANOVA F (categorical)',
      'Chi-squared (cat-cat)',
      'Point-Biserial (binary)',
      '',
      'alpha = 0.05 filter',
      '-> Claim Objects'],
     C['box_blue'])

# Devil's Advocate
rbox(ax, 6.0, 1.3, 2.8, 4.2,
     "DEVIL'S ADVOCATE",
     ['Battery 1: Simpson Scan',
      'Battery 2: Partial Corr.',
      'Battery 3: Bootstrap',
      'Battery 4: Leakage',
      '',
      'Utility: MINIMIZE',
      'claim survival'],
     C['box_red'])

# Referee
rbox(ax, 9.4, 1.3, 2.8, 4.2,
     'REFEREE AGENT',
     ['Evidence weighting',
      'Verdict assignment',
      '',
      'ROBUST',
      "SIMPSON'S PARADOX",
      'SPURIOUS',
      'Recommendation'],
     C['box_green'])

# Output
rbox(ax, 12.7, 2.3, 2.0, 2.3,
     'REPORT',
     ['HTML', 'Markdown', 'JSON'],
     C['box_dark'])

# Arrows
arrow(ax, 2.2, 3.5, 2.6,  'Dataset',  '#546E7A')
arrow(ax, 5.4, 3.5, 6.0,  'Claims',   '#546E7A')
arrow(ax, 8.8, 3.5, 9.4,  'Dossier',  '#546E7A')
arrow(ax, 12.2, 3.5, 12.7, 'Verdict', '#546E7A')

# Utility labels below each agent box
for x_c, label, col in [
    (4.0,  'Utility: maximize\nsignificant findings', C['box_blue']),
    (7.4,  'Utility: MINIMIZE\nclaim survival',        C['box_red']),
    (10.8, 'Utility: calibrated\naccuracy',            C['box_green']),
]:
    ax.text(x_c, 0.9, label, ha='center', va='top', fontsize=9.5,
            color=col, style='italic', fontweight='bold', zorder=4)

# Title
ax.set_title(
    'Fig. 4  —  ADAA System Architecture: Three Rational Agents with Opposing Utility Functions',
    fontsize=14, fontweight='bold', pad=14, color=C['text'])

save(fig, 'fig4_architecture.png')


# =============================================================================
# FIG 5  Bootstrap stability: 3 feature-income pairs
# =============================================================================
print("[5/5] Bootstrap stability...")
adult = pd.read_csv(ADULT)
np.random.seed(42)
B = 150

def boot_ci(x, y, B=150):
    n = len(x)
    return np.array([
        np.corrcoef(x[np.random.choice(n, n, replace=True)],
                    y[np.random.choice(n, n, replace=True)])[0, 1]
        for _ in range(B)
    ])

edu_boot = boot_ci(adult['education_num'].values, adult['income_above_50k'].values)
hrs_boot = boot_ci(adult['hours_per_week'].values, adult['income_above_50k'].values)
age_boot = boot_ci(adult['age'].values,            adult['income_above_50k'].values)

configs = [
    (edu_boot, 'education_num -> income', C['robust'],     'ROBUST & VERIFIED'),
    (hrs_boot, 'hours_per_week -> income', C['confounded'], 'SPURIOUS / CONFOUNDED'),
    (age_boot, 'age -> income',            C['paradox'],    "SIMPSON'S PARADOX\n+ UNSTABLE"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

for ax, (boot, title, color, verdict) in zip(axes, configs):
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    mn            = boot.mean()

    n_bins = 28
    counts, edges = np.histogram(boot, bins=n_bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges),
           color=color, alpha=0.80, zorder=3, linewidth=0, align='edge')

    ax.axvline(mn,    color=C['text'],  linestyle='-',  linewidth=2.0, zorder=5, label=f'Mean: {mn:.3f}')
    ax.axvline(ci_lo, color='black',    linestyle='--', linewidth=1.8, zorder=5, label=f'CI low:  {ci_lo:.3f}')
    ax.axvline(ci_hi, color='black',    linestyle=':',  linewidth=1.8, zorder=5, label=f'CI high: {ci_hi:.3f}')
    ax.axvline(0,     color='#C62828',  linestyle='-',  linewidth=1.4, zorder=4, alpha=0.7, label='r = 0')

    # Shade CI region
    ax.axvspan(ci_lo, ci_hi, alpha=0.12, color=color, zorder=2)

    # Verdict badge
    badge_col = C['robust'] if 'ROBUST' in verdict else (C['confounded'] if 'CONF' in verdict else C['paradox'])
    ax.text(0.5, 0.97, verdict, transform=ax.transAxes,
            ha='center', va='top', fontsize=10, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=badge_col, edgecolor='none', alpha=0.92))

    # Zero-crossing warning
    if ci_lo < 0 < ci_hi:
        ax.text(0.5, 0.78, 'CI crosses zero!\nUnstable finding', transform=ax.transAxes,
                ha='center', va='top', fontsize=9.5, color=C['confounded'],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor=C['paradox']))

    ax.set_xlabel('Bootstrap Pearson r', fontsize=12)
    ax.set_ylabel('Count' if ax == axes[0] else '', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
    ax.set_axisbelow(True)

fig.suptitle(
    'Fig. 5  —  Battery 3: Bootstrap Stability Analysis  (B = 150 resamples, 95% CI)',
    fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=3)
save(fig, 'fig5_bootstrap_stability.png')

print("\nAll 5 figures saved to docs/figures/ at 300 DPI")

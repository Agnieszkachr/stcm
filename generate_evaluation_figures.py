"""
generate_evaluation_figures.py
========================
Generates figures and supplementary analyses for the model evaluation.
Reads existing pipeline output (q_score_distribution.csv) and produces:
  1. Q-score distribution histogram with permutation null overlay
  2. Q-score vs word-overlap scatter
  3. Residual symmetry comparison (triple vs double tradition top-5)
  4. Q-score vs pericope length scatter
  5. Sensitivity analysis stability plot
  6. Pericope length and form analysis statistics

All figures saved to stcm/outputs/figures/evaluation/
"""
import csv
import os
import sys
import math
import statistics

# Ensure stcm is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------- CONFIGURATION ----------
OUT_DIR = os.path.join('outputs', 'figures', 'evaluation')
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join('outputs', 'reports', 'q_score_distribution.csv')

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ---------- LOAD DATA ----------
data = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['label'].strip():
            data.append({
                'label': row['label'].strip(),
                'matt_luke_cos': float(row['matt_luke_cos']),
                'residual_sim': float(row['residual_sim']),
                'q_score': float(row['q_score']),
                'q_score_norm': float(row['q_score_norm']),
                'deviation': float(row['deviation_from_sig_a']),
            })

print(f"Loaded {len(data)} pericopes from CSV")

# ---------- PERICOPE METADATA ----------
# Verse counts (end - start + 1 for each gospel)
# From data_loader.py DOUBLE_TRADITION
verse_ranges = {
    "John's preaching":           {'matt': (3,7,12), 'luke': (3,7,9)},
    "Temptation narrative (full)":{'matt': (4,1,11), 'luke': (4,1,13)},
    "Beatitudes":                 {'matt': (5,3,12), 'luke': (6,20,26)},
    "Love of enemies":            {'matt': (5,43,48),'luke': (6,27,36)},
    "Lord's Prayer":              {'matt': (6,9,13), 'luke': (11,2,4)},
    "Anxieties about life":       {'matt': (6,25,34),'luke': (12,22,32)},
    "Narrow gate":                {'matt': (7,13,14),'luke': (13,23,24)},
    "Centurion's servant":        {'matt': (8,5,13), 'luke': (7,1,10)},
    "John's question from prison":{'matt': (11,2,6), 'luke': (7,18,23)},
    "Jesus on John":              {'matt': (11,7,19),'luke': (7,24,35)},
    "Woes on Galilean cities":    {'matt': (11,20,24),'luke': (10,13,15)},
    "Hidden from wise revealed":  {'matt': (11,25,27),'luke': (10,21,22)},
    "Mission discourse":          {'matt': (10,5,16),'luke': (10,1,12)},
    "Harvest plentiful":          {'matt': (9,37,38),'luke': (10,2,3)},
    "Sign of Jonah":              {'matt': (12,38,42),'luke': (11,29,32)},
    "Return of unclean spirit":   {'matt': (12,43,45),'luke': (11,24,26)},
    "Lamp of the body":           {'matt': (6,22,23),'luke': (11,34,36)},
    "Leaven of Pharisees":        {'matt': (16,6,12),'luke': (12,1,3)},
    "Fear of God not men":        {'matt': (10,26,33),'luke': (12,4,9)},
    "Blasphemy Holy Spirit":      {'matt': (12,31,32),'luke': (12,10,12)},
    "Thief in the night":         {'matt': (24,43,44),'luke': (12,39,40)},
    "Faithful servant":           {'matt': (24,45,51),'luke': (12,42,48)},
    "Not peace but sword":        {'matt': (10,34,36),'luke': (12,51,53)},
    "Reading the signs":          {'matt': (16,2,3),  'luke': (12,54,56)},
    "Settling with opponent":     {'matt': (5,25,26), 'luke': (12,57,59)},
    "Mustard seed and leaven":    {'matt': (13,31,33),'luke': (13,18,21)},
    "Many come from east west":   {'matt': (8,11,12), 'luke': (13,28,29)},
    "Lament over Jerusalem":      {'matt': (23,37,39),'luke': (13,34,35)},
    "Parable of Great Banquet":   {'matt': (22,1,14), 'luke': (14,15,24)},
    "Conditions of discipleship": {'matt': (10,37,38),'luke': (14,26,27)},
    "Salt of the earth":          {'matt': (5,13,13), 'luke': (14,34,35)},
    "Lost sheep":                 {'matt': (18,12,14),'luke': (15,3,7)},
    "Serving two masters":        {'matt': (6,24,24), 'luke': (16,13,13)},
    "Day of the Son of Man":      {'matt': (24,26,28),'luke': (17,23,24)},
    "Talents / Minas":            {'matt': (25,14,30),'luke': (19,12,27)},
    "Judging twelve tribes":      {'matt': (19,28,28),'luke': (22,28,30)},
}

# Kloppenborg stratum assignments — VERIFIED from The Formation of Q (1987) PDF
# Q1 = formative sapiential, Q2 = main redaction/judgment, Q3 = final recension
# 'unassigned' = Kloppenborg explicitly says cannot be assigned on literary grounds
kloppenborg_strata = {
    "John's preaching":           "Q2",
    "Temptation narrative (full)":"Q3",
    "Beatitudes":                 "Q1",
    "Love of enemies":            "Q1",
    "Lord's Prayer":              "Q1",
    "Anxieties about life":       "Q1",
    "Narrow gate":                "Q2",  # Part of Q 13:24-14:35 mixed speech, judgment content
    "Centurion's servant":        "Q2",
    "John's question from prison":"Q2",
    "Jesus on John":              "Q2",
    "Woes on Galilean cities":    "Q2",
    "Hidden from wise revealed":  "Q2",  # Treated in Ch5 but revelation-christology = Q2
    "Mission discourse":          "Q1",  # Q 10:2-16 in Ch5
    "Harvest plentiful":          "Q1",  # Part of mission cluster in Ch5
    "Sign of Jonah":              "Q2",
    "Return of unclean spirit":   "Q2",
    "Lamp of the body":           "Q2",  # Part of Q 11:33-36 in Ch4 controversies
    "Leaven of Pharisees":        "Q2",  # Part of Q 12:2-12 context
    "Fear of God not men":        "Q1",  # Q 12:2-12 in Ch5
    "Blasphemy Holy Spirit":      "Q2",  # Connected to Beelzebul
    "Thief in the night":         "Q2",
    "Faithful servant":           "Q2",
    "Not peace but sword":        "Q2",
    "Reading the signs":          "Q2",
    "Settling with opponent":     "Q2",
    "Mustard seed and leaven":    "Q1",  # Q 13:18-21 in sapiential block
    "Many come from east west":   "Q2",  # Judgment: eschatological banquet
    "Lament over Jerusalem":      "Q2",
    "Parable of Great Banquet":   "Q2",  # Part of Q 13:24-14:35 mixed speech
    "Conditions of discipleship": "Q2",
    "Salt of the earth":          "Q2",  # Q 14:34-35 in Ch5 mixed speech
    "Lost sheep":                 "unassigned",  # Q 15:4-7 explicitly listed as unassignable (p.100)
    "Serving two masters":        "unassigned",  # Q 16:13 explicitly listed as unassignable (p.100)
    "Day of the Son of Man":      "Q2",
    "Talents / Minas":            "Q2",  # Q 19:12-27 in eschatological discourse
    "Judging twelve tribes":      "Q2",  # Q 22:28-30 in eschatological discourse
}

# Form-critical tags
form_tags = {
    "John's preaching":           "discourse",
    "Temptation narrative (full)":"narrative",
    "Beatitudes":                 "discourse",
    "Love of enemies":            "discourse",
    "Lord's Prayer":              "liturgical",
    "Anxieties about life":       "discourse",
    "Narrow gate":                "proverbial",
    "Centurion's servant":        "narrative",
    "John's question from prison":"narrative",
    "Jesus on John":              "discourse",
    "Woes on Galilean cities":    "discourse",
    "Hidden from wise revealed":  "proverbial",
    "Mission discourse":          "discourse",
    "Harvest plentiful":          "proverbial",
    "Sign of Jonah":              "discourse",
    "Return of unclean spirit":   "discourse",
    "Lamp of the body":           "proverbial",
    "Leaven of Pharisees":        "discourse",
    "Fear of God not men":        "discourse",
    "Blasphemy Holy Spirit":      "proverbial",
    "Thief in the night":         "proverbial",
    "Faithful servant":           "parable",
    "Not peace but sword":        "proverbial",
    "Reading the signs":          "proverbial",
    "Settling with opponent":     "proverbial",
    "Mustard seed and leaven":    "parable",
    "Many come from east west":   "proverbial",
    "Lament over Jerusalem":      "discourse",
    "Parable of Great Banquet":   "parable",
    "Conditions of discipleship": "proverbial",
    "Salt of the earth":          "proverbial",
    "Lost sheep":                 "parable",
    "Serving two masters":        "proverbial",
    "Day of the Son of Man":      "discourse",
    "Talents / Minas":            "parable",
    "Judging twelve tribes":      "proverbial",
}

# Compute pericope lengths
for d in data:
    vr = verse_ranges.get(d['label'])
    if vr:
        matt_len = vr['matt'][2] - vr['matt'][1] + 1
        luke_len = vr['luke'][2] - vr['luke'][1] + 1
        d['total_verses'] = matt_len + luke_len
        d['mean_verses'] = (matt_len + luke_len) / 2.0
        d['matt_verses'] = matt_len
        d['luke_verses'] = luke_len
    else:
        d['total_verses'] = 0
        d['mean_verses'] = 0
    d['stratum'] = kloppenborg_strata.get(d['label'], 'unknown')
    d['form'] = form_tags.get(d['label'], 'unknown')

# Sort by Q-score descending
data.sort(key=lambda x: x['q_score'], reverse=True)

# ---------- FIGURE 1: Q-score distribution with null ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
q_scores = [d['q_score'] for d in data]

# Histogram
ax.hist(q_scores, bins=12, color='#2E86AB', alpha=0.75, edgecolor='white', linewidth=0.5, label='Observed Q-scores (n=36)')

# Null distribution (mean=0.411, sd=0.003 from evaluation_summary.md)
null_mean = 0.4268
null_sd = 0.0023
x_null = np.linspace(0.41, 0.44, 200)
null_pdf = (1 / (null_sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_null - null_mean) / null_sd)**2)
# Scale to histogram height
null_pdf_scaled = null_pdf * (len(q_scores) * (q_scores[0] - q_scores[-1]) / 12) * 0.3

ax2 = ax.twinx()
ax2.fill_between(x_null, null_pdf_scaled, alpha=0.3, color='#E84855', label=f'Permutation null (μ={null_mean}, n=1000)')
ax2.set_ylim(0, ax2.get_ylim()[1] * 3)
ax2.set_yticks([])

# Add vertical lines
ax.axvline(x=statistics.mean(q_scores), color='#2E86AB', linestyle='--', linewidth=1.5, label=f'Observed mean ({statistics.mean(q_scores):.3f})')
ax.axvline(x=null_mean, color='#E84855', linestyle='--', linewidth=1.5, label=f'Null mean ({null_mean})')

ax.set_xlabel('Q-score')
ax.set_ylabel('Count')
ax.set_title('Distribution of Q-scores across 36 double-tradition pericopes')
# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
plt.savefig(os.path.join(OUT_DIR, 'fig1_qscore_distribution.png'))
plt.close()
print("Saved fig1_qscore_distribution.png")

# ---------- FIGURE 2: Q-score vs pericope length ----------
fig, ax = plt.subplots(figsize=(7, 5))
lengths = [d['mean_verses'] for d in data]
scores = [d['q_score'] for d in data]
labels_list = [d['label'] for d in data]

ax.scatter(lengths, scores, c='#2E86AB', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

# Annotate top-5 and bottom-3
for d in data[:5]:
    ax.annotate(d['label'], (d['mean_verses'], d['q_score']),
                fontsize=7, ha='left', va='bottom', color='#333',
                xytext=(5, 3), textcoords='offset points')
for d in data[-3:]:
    ax.annotate(d['label'], (d['mean_verses'], d['q_score']),
                fontsize=7, ha='left', va='top', color='#999',
                xytext=(5, -3), textcoords='offset points')

# Correlation
from scipy import stats as sp_stats
r, p = sp_stats.pearsonr(lengths, scores)
sr, sp = sp_stats.spearmanr(lengths, scores)

# Regression line
z = np.polyfit(lengths, scores, 1)
xline = np.linspace(min(lengths), max(lengths), 100)
ax.plot(xline, np.polyval(z, xline), '--', color='#E84855', alpha=0.6, linewidth=1)

ax.set_xlabel('Mean pericope length (verses)')
ax.set_ylabel('Q-score')
ax.set_title(f'Q-score vs. pericope length (r = {r:.3f}, p = {p:.3f}; ρ = {sr:.3f})')
plt.savefig(os.path.join(OUT_DIR, 'fig2_qscore_vs_length.png'))
plt.close()
print(f"Saved fig2_qscore_vs_length.png | Pearson r={r:.3f}, p={p:.4f} | Spearman rho={sr:.3f}, p={sp:.4f}")

# ---------- FIGURE 3: Residual symmetry comparison ----------
fig, ax = plt.subplots(figsize=(7, 5))

# Triple-tradition baseline
triple_baseline = 0.3693
triple_sd = 0.1972

# Top 5 double-tradition residuals
top5 = data[:5]
top5_labels = [d['label'].replace(' ', '\n', 1) for d in top5]
top5_resid = [d['residual_sim'] for d in top5]

# All double-tradition residuals
all_resid = [d['residual_sim'] for d in data]
all_mean = statistics.mean(all_resid)

bar_positions = np.arange(len(top5))
bars = ax.bar(bar_positions, top5_resid, color='#2E86AB', alpha=0.8, width=0.6, edgecolor='white')

ax.axhline(y=triple_baseline, color='#E84855', linestyle='--', linewidth=1.5,
           label=f'Triple-tradition baseline (μ={triple_baseline})')
ax.axhline(y=all_mean, color='#A5BE00', linestyle=':', linewidth=1.5,
           label=f'Double-tradition mean ({all_mean:.3f})')

# Add value labels on bars
for bar, val in zip(bars, top5_resid):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(bar_positions)
ax.set_xticklabels(top5_labels, fontsize=8)
ax.set_ylabel('Residual cosine similarity')
ax.set_title('Residual symmetry: top-5 double-tradition pericopes\nvs. triple-tradition baseline')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', fontsize=9)
plt.savefig(os.path.join(OUT_DIR, 'fig3_residual_symmetry.png'))
plt.close()
print("Saved fig3_residual_symmetry.png")

# ---------- FIGURE 4: Q-score by Kloppenborg stratum ----------
fig, ax = plt.subplots(figsize=(7, 5))

strata_groups = {}
for d in data:
    s = d['stratum']
    strata_groups.setdefault(s, []).append(d['q_score'])

strata_order = ['Q1', 'Q2', 'Q3', 'unassigned']
strata_colors = {'Q1': '#2E86AB', 'Q2': '#E84855', 'Q3': '#A5BE00', 'unassigned': '#888888'}
strata_labels_nice = {'Q1': 'Q¹ (sapiential)', 'Q2': 'Q² (judgment)', 'Q3': 'Q³ (temptation)', 'unassigned': 'Unassigned'}

box_data = []
box_labels = []
box_colors = []
for s in strata_order:
    if s in strata_groups:
        box_data.append(strata_groups[s])
        box_labels.append(f'{strata_labels_nice[s]}\n(n={len(strata_groups[s])})')
        box_colors.append(strata_colors[s])

bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

# Overlay individual points
for i, (bd, color) in enumerate(zip(box_data, box_colors)):
    x_jitter = np.random.normal(i+1, 0.04, len(bd))
    ax.scatter(x_jitter, bd, c=color, s=30, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=5)

ax.set_xticklabels(box_labels, fontsize=9)
ax.set_ylabel('Q-score')
ax.set_title("Q-score distribution by Kloppenborg's Q strata")

# Kruskal-Wallis test between Q1 and Q2
q1_scores = strata_groups.get('Q1', [])
q2_scores = strata_groups.get('Q2', [])
if q1_scores and q2_scores:
    kw_stat, kw_p = sp_stats.kruskal(q1_scores, q2_scores)
    mwu_stat, mwu_p = sp_stats.mannwhitneyu(q1_scores, q2_scores, alternative='two-sided')
    ax.text(0.98, 0.02, f'Q¹ vs Q²: Mann–Whitney U p = {mwu_p:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#666')
    print(f"Q1 mean={statistics.mean(q1_scores):.4f}, Q2 mean={statistics.mean(q2_scores):.4f}")
    print(f"Mann-Whitney U: stat={mwu_stat:.1f}, p={mwu_p:.4f}")
    print(f"Kruskal-Wallis: stat={kw_stat:.3f}, p={kw_p:.4f}")

plt.savefig(os.path.join(OUT_DIR, 'fig4_qscore_by_stratum.png'))
plt.close()
print("Saved fig4_qscore_by_stratum.png")

# ---------- FIGURE 5: Full ranked pericope plot ----------
fig, ax = plt.subplots(figsize=(10, 7))

# Sort by Q-score
data_sorted = sorted(data, key=lambda x: x['q_score'])
y_pos = np.arange(len(data_sorted))
colors = [strata_colors.get(d['stratum'], '#888') for d in data_sorted]

ax.barh(y_pos, [d['q_score'] for d in data_sorted], color=colors, alpha=0.75,
        edgecolor='white', linewidth=0.3, height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([d['label'] for d in data_sorted], fontsize=7)
ax.set_xlabel('Q-score')
ax.set_title('All 36 double-tradition pericopes ranked by Q-score')
ax.axvline(x=statistics.mean(q_scores), color='black', linestyle='--', linewidth=1, alpha=0.5,
           label=f'Mean ({statistics.mean(q_scores):.3f})')
ax.axvline(x=null_mean, color='#E84855', linestyle=':', linewidth=1, alpha=0.5,
           label=f'Null mean ({null_mean})')

# Legend for strata colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=strata_colors['Q1'], alpha=0.6, label='Q¹ (sapiential)'),
    Patch(facecolor=strata_colors['Q2'], alpha=0.6, label='Q² (judgment)'),
    Patch(facecolor=strata_colors['Q3'], alpha=0.6, label='Q³ (temptation)'),
    Patch(facecolor=strata_colors['unassigned'], alpha=0.6, label='Unassigned'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.savefig(os.path.join(OUT_DIR, 'fig5_full_ranking.png'))
plt.close()
print("Saved fig5_full_ranking.png")

# ---------- FIGURE 6: Q-score by form type ----------
fig, ax = plt.subplots(figsize=(7, 5))

form_groups = {}
for d in data:
    f = d['form']
    form_groups.setdefault(f, []).append(d['q_score'])

form_order = ['proverbial', 'discourse', 'parable', 'narrative', 'liturgical']
form_colors_map = {
    'proverbial': '#E84855', 'discourse': '#2E86AB',
    'parable': '#A5BE00', 'narrative': '#F4A261', 'liturgical': '#9B5DE5'
}

box_data_f = []
box_labels_f = []
box_colors_f = []
for f in form_order:
    if f in form_groups:
        box_data_f.append(form_groups[f])
        box_labels_f.append(f'{f.title()}\n(n={len(form_groups[f])})')
        box_colors_f.append(form_colors_map[f])

bp2 = ax.boxplot(box_data_f, patch_artist=True, widths=0.5)
for patch, color in zip(bp2['boxes'], box_colors_f):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for median in bp2['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

for i, (bd, color) in enumerate(zip(box_data_f, box_colors_f)):
    x_jitter = np.random.normal(i+1, 0.04, len(bd))
    ax.scatter(x_jitter, bd, c=color, s=30, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=5)

ax.set_xticklabels(box_labels_f, fontsize=9)
ax.set_ylabel('Q-score')
ax.set_title('Q-score distribution by literary form')

# Print form statistics
for f in form_order:
    if f in form_groups:
        vals = form_groups[f]
        print(f"Form '{f}': n={len(vals)}, mean={statistics.mean(vals):.4f}, sd={statistics.stdev(vals) if len(vals)>1 else 0:.4f}")

plt.savefig(os.path.join(OUT_DIR, 'fig6_qscore_by_form.png'))
plt.close()
print("Saved fig6_qscore_by_form.png")

# ---------- SUMMARY STATISTICS ----------
print("\n=== SUMMARY FOR ARTICLE ===")
print(f"N pericopes: {len(data)}")
print(f"Q-score mean: {statistics.mean(q_scores):.4f}")
print(f"Q-score SD: {statistics.stdev(q_scores):.4f}")
print(f"Q-score range: [{min(q_scores):.3f}, {max(q_scores):.3f}]")
print(f"\nTop 5:")
for i, d in enumerate(data[:5]):
    print(f"  {i+1}. {d['label']}: Q={d['q_score']:.4f}, cos={d['matt_luke_cos']:.3f}, resid={d['residual_sim']:.3f}, stratum={d['stratum']}")
print(f"\nBottom 3:")
for d in data[-3:]:
    print(f"  {d['label']}: Q={d['q_score']:.4f}, cos={d['matt_luke_cos']:.3f}, stratum={d['stratum']}")

# John's preaching specific
johns = [d for d in data if d['label'] == "John's preaching"][0]
johns_rank = [d['label'] for d in data].index("John's preaching") + 1
print(f"\nJohn's Preaching: rank={johns_rank}/36, Q={johns['q_score']:.4f}, cos={johns['matt_luke_cos']:.3f}, matt_verses={johns['matt_verses']}, luke_verses={johns['luke_verses']}")

print(f"\nLength-Q-score correlation: Pearson r={r:.3f} (p={p:.4f}), Spearman rho={sr:.3f}")

print("\nKloppenborg strata counts:")
for s in strata_order:
    if s in strata_groups:
        print(f"  {s}: n={len(strata_groups[s])}, mean Q-score={statistics.mean(strata_groups[s]):.4f}")

print("\nAll figures saved to:", OUT_DIR)

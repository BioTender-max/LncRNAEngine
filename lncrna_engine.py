"""
LncRNAEngine: Long Non-Coding RNA Regulatory Analysis Pipeline
- lncRNA-mRNA co-expression network (Pearson correlation, FDR)
- RNA-binding protein (RBP) motif enrichment
- ceRNA (competing endogenous RNA) network via shared miRNA targets
- Nuclear vs cytoplasmic localization prediction
- lncRNA conservation scoring
"""

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Parameters ───────────────────────────────────────────────────────────────
N_SAMPLES  = 100
N_LNCRNA   = 2000
N_MRNA     = 5000
N_MIRNA    = 300
N_RBP      = 50

print("=" * 65)
print("  LncRNAEngine: Long Non-Coding RNA Regulatory Analysis")
print("=" * 65)

# ─── Simulate expression data ─────────────────────────────────────────────────
print("\n[1] Simulating lncRNA and mRNA expression data...")

# lncRNA expression (log-normal, many lowly expressed)
lnc_expr = np.random.lognormal(mean=1.5, sigma=1.8, size=(N_SAMPLES, N_LNCRNA))
lnc_log  = np.log2(lnc_expr + 1)

# mRNA expression
mrna_expr = np.random.lognormal(mean=3.0, sigma=1.5, size=(N_SAMPLES, N_MRNA))
mrna_log  = np.log2(mrna_expr + 1)

# Inject co-expression signal: first 200 lncRNAs correlated with first 500 mRNAs
for i in range(200):
    partner = i % 500
    noise   = np.random.normal(0, 0.3, N_SAMPLES)
    lnc_log[:, i] = mrna_log[:, partner] * np.random.uniform(0.6, 1.0) + noise

print(f"    lncRNAs: {N_LNCRNA}, mRNAs: {N_MRNA}, Samples: {N_SAMPLES}")

# ─── 1. Co-expression: Pearson r + BH FDR ────────────────────────────────────
print("\n[2] Computing lncRNA-mRNA co-expression (subset)...")

# Use top 200 lncRNAs (by variance) × top 500 mRNAs for speed
lnc_var  = np.var(lnc_log, axis=0)
mrna_var = np.var(mrna_log, axis=0)
top_lnc  = np.argsort(lnc_var)[::-1][:200]
top_mrna = np.argsort(mrna_var)[::-1][:500]

lnc_sub  = lnc_log[:, top_lnc]
mrna_sub = mrna_log[:, top_mrna]

# Vectorized Pearson r
def pearson_matrix(A, B):
    """A: (n,p), B: (n,q) → r: (p,q)"""
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)
    num = A_c.T @ B_c
    denom = np.sqrt((A_c**2).sum(axis=0)[:, None] * (B_c**2).sum(axis=0)[None, :])
    return num / (denom + 1e-12)

r_mat = pearson_matrix(lnc_sub, mrna_sub)   # (200, 500)

# t-statistic and p-value
n = N_SAMPLES
t_mat = r_mat * np.sqrt(n - 2) / np.sqrt(np.maximum(1 - r_mat**2, 1e-12))
pval_mat = 2 * stats.t.sf(np.abs(t_mat), df=n - 2)

# BH FDR on flattened
def bh_fdr(pv):
    n_p = len(pv)
    order = np.argsort(pv)
    ranks = np.empty(n_p); ranks[order] = np.arange(1, n_p + 1)
    fdr = np.minimum(pv * n_p / ranks, 1.0)
    fdr_adj = np.minimum.accumulate(fdr[order][::-1])[::-1]
    result = np.empty(n_p); result[order] = fdr_adj
    return result

pval_flat = pval_mat.ravel()
fdr_flat  = bh_fdr(pval_flat)
fdr_mat   = fdr_flat.reshape(r_mat.shape)

sig_coexpr = (fdr_mat < 0.05) & (np.abs(r_mat) > 0.5)
n_coexpr   = int(np.sum(sig_coexpr))
print(f"    Significant co-expression pairs (FDR<0.05, |r|>0.5): {n_coexpr}")

# Top 50 pairs for heatmap
flat_r   = r_mat.ravel()
flat_fdr = fdr_mat.ravel()
sig_idx  = np.where(sig_coexpr.ravel())[0]
if len(sig_idx) >= 50:
    top50_idx = sig_idx[np.argsort(np.abs(flat_r[sig_idx]))[::-1][:50]]
else:
    top50_idx = np.argsort(np.abs(flat_r))[::-1][:50]

top50_lnc_i  = top50_idx // 500
top50_mrna_i = top50_idx %  500
uniq_lnc  = np.unique(top50_lnc_i)[:20]
uniq_mrna = np.unique(top50_mrna_i)[:20]
heatmap_r = r_mat[np.ix_(uniq_lnc, uniq_mrna)]

# ─── 2. RBP motif enrichment ──────────────────────────────────────────────────
print("\n[3] RBP motif enrichment analysis...")

# Simulate lncRNA sequence lengths and motif hit counts
lnc_lengths = np.random.randint(500, 10000, size=N_LNCRNA)
# For each RBP, simulate motif hits (Poisson, rate proportional to length)
rbp_rates   = np.random.uniform(0.001, 0.01, size=N_RBP)  # hits per bp
rbp_hits    = np.zeros((N_LNCRNA, N_RBP), dtype=int)
for j in range(N_RBP):
    expected = lnc_lengths * rbp_rates[j]
    rbp_hits[:, j] = np.random.poisson(expected)

# Background: shuffle lengths
bg_lengths  = np.random.randint(200, 3000, size=N_LNCRNA)
rbp_hits_bg = np.zeros((N_LNCRNA, N_RBP), dtype=int)
for j in range(N_RBP):
    expected_bg = bg_lengths * rbp_rates[j]
    rbp_hits_bg[:, j] = np.random.poisson(expected_bg)

# Enrichment: fold change of mean hits
rbp_fc   = (np.mean(rbp_hits, axis=0) + 1e-3) / (np.mean(rbp_hits_bg, axis=0) + 1e-3)
rbp_pval = np.array([stats.mannwhitneyu(rbp_hits[:, j], rbp_hits_bg[:, j],
                                         alternative='greater').pvalue
                      for j in range(N_RBP)])
rbp_fdr  = bh_fdr(rbp_pval)
top10_rbp = np.argsort(rbp_fc)[::-1][:10]
print(f"    RBPs tested: {N_RBP}, significant (FDR<0.05): {int(np.sum(rbp_fdr<0.05))}")

# ─── 3. ceRNA network: Jaccard similarity of shared miRNA targets ─────────────
print("\n[4] Computing ceRNA network (Jaccard similarity)...")

# Simulate miRNA target sets for lncRNAs and mRNAs
lnc_mirna_targets  = np.random.randint(0, 2, size=(N_LNCRNA, N_MIRNA)).astype(bool)
mrna_mirna_targets = np.random.randint(0, 2, size=(N_MRNA,   N_MIRNA)).astype(bool)

# Jaccard for top 100 lncRNAs × top 200 mRNAs
n_lnc_sub  = 100
n_mrna_sub = 200
lnc_t  = lnc_mirna_targets[:n_lnc_sub]
mrna_t = mrna_mirna_targets[:n_mrna_sub]

# Vectorized Jaccard
inter = lnc_t.astype(int) @ mrna_t.astype(int).T          # (100, 200)
union = (lnc_t.sum(axis=1)[:, None] + mrna_t.sum(axis=1)[None, :]
         - inter)
jaccard = inter / (union + 1e-6)

top20_cerna_flat = np.argsort(jaccard.ravel())[::-1][:20]
top20_lnc_c  = top20_cerna_flat // n_mrna_sub
top20_mrna_c = top20_cerna_flat %  n_mrna_sub
uniq_lnc_c   = np.unique(top20_lnc_c)[:10]
uniq_mrna_c  = np.unique(top20_mrna_c)[:10]
cerna_heatmap = jaccard[np.ix_(uniq_lnc_c, uniq_mrna_c)]
print(f"    Max Jaccard ceRNA score: {jaccard.max():.4f}")

# ─── 4. Nuclear vs cytoplasmic localization ───────────────────────────────────
print("\n[5] Predicting lncRNA localization...")

# Simulate sequence features
gc_content = np.random.beta(5, 5, size=N_LNCRNA)  # GC fraction
au_content = 1 - gc_content
lengths_loc = lnc_lengths.copy()

# Nuclear: high GC (>0.5) AND long (>2000 bp)
# Cytoplasmic: short (<1000 bp) OR AU-rich (AU>0.55)
nuclear_score = (gc_content > 0.5).astype(float) + (lengths_loc > 2000).astype(float)
nuclear_pred  = nuclear_score >= 1.5
n_nuclear     = int(np.sum(nuclear_pred))
n_cytoplasmic = N_LNCRNA - n_nuclear
print(f"    Nuclear: {n_nuclear}, Cytoplasmic: {n_cytoplasmic}")

# ─── 5. Conservation scoring ──────────────────────────────────────────────────
print("\n[6] Conservation analysis (PhyloP scores)...")

# Simulate PhyloP scores; expressed lncRNAs slightly more conserved
mean_expr_lnc = np.mean(lnc_log, axis=0)
expressed     = mean_expr_lnc > np.median(mean_expr_lnc)
phylop_scores = np.where(expressed,
                         np.random.normal(1.2, 0.8, N_LNCRNA),
                         np.random.normal(0.3, 0.8, N_LNCRNA))
t_cons, p_cons = stats.ttest_ind(phylop_scores[expressed], phylop_scores[~expressed])
print(f"    PhyloP expressed vs non-expressed: t={t_cons:.3f}, p={p_cons:.2e}")

# Hub lncRNAs: most co-expression connections
lnc_degree = np.sum(sig_coexpr, axis=1)  # (200,)
# Pad to N_LNCRNA
lnc_degree_full = np.zeros(N_LNCRNA)
lnc_degree_full[top_lnc] = lnc_degree
top20_hub = np.argsort(lnc_degree_full)[::-1][:20]

# ─── Dashboard ────────────────────────────────────────────────────────────────
print("\n[7] Generating dashboard...")

DARK_BG  = '#0a0a0a'
PANEL_BG = '#111111'
WHITE    = '#ffffff'
ACCENT1  = '#00d4ff'
ACCENT2  = '#ff6b6b'
ACCENT3  = '#51cf66'
ACCENT4  = '#ffd43b'
ACCENT5  = '#cc5de8'
GRID_CLR = '#2a2a2a'

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(DARK_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    if title:  ax.set_title(title,  color=WHITE, fontsize=9, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=WHITE, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=WHITE, fontsize=8)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.7)

# P1: Co-expression heatmap (top 50 pairs → 20×20 submatrix)
ax1 = fig.add_subplot(gs[0, 0])
hm1 = ax1.imshow(heatmap_r, aspect='auto', cmap='RdBu_r',
                  vmin=-1, vmax=1, interpolation='nearest')
cb1 = plt.colorbar(hm1, ax=ax1, fraction=0.04, pad=0.02)
cb1.ax.tick_params(colors=WHITE, labelsize=7)
style_ax(ax1, 'Co-expression Network (top pairs)', 'mRNA', 'lncRNA')
ax1.grid(False)

# P2: RBP motif enrichment bar (top 10)
ax2 = fig.add_subplot(gs[0, 1])
top10_fc   = rbp_fc[top10_rbp]
top10_labs = [f'RBP_{i}' for i in top10_rbp]
bar_cols   = plt.cm.plasma(np.linspace(0.2, 0.9, 10))
ax2.barh(range(10), top10_fc[::-1], color=bar_cols, edgecolor='none')
ax2.set_yticks(range(10))
ax2.set_yticklabels(top10_labs[::-1], fontsize=7, color=WHITE)
ax2.axvline(1.0, color=ACCENT2, linestyle='--', linewidth=1.2)
style_ax(ax2, 'RBP Motif Enrichment (top 10)', 'Fold Change', '')
ax2.grid(True, axis='x', color=GRID_CLR, linewidth=0.5)

# P3: ceRNA network heatmap
ax3 = fig.add_subplot(gs[0, 2])
hm3 = ax3.imshow(cerna_heatmap, aspect='auto', cmap='YlOrRd',
                  interpolation='nearest')
cb3 = plt.colorbar(hm3, ax=ax3, fraction=0.04, pad=0.02)
cb3.ax.tick_params(colors=WHITE, labelsize=7)
style_ax(ax3, 'ceRNA Network (Jaccard similarity)', 'mRNA', 'lncRNA')
ax3.grid(False)

# P4: Nuclear vs cytoplasmic bar
ax4 = fig.add_subplot(gs[1, 0])
cats   = ['Nuclear', 'Cytoplasmic']
counts = [n_nuclear, n_cytoplasmic]
bar4   = ax4.bar(cats, counts, color=[ACCENT1, ACCENT2], edgecolor='none', width=0.5)
for b, c in zip(bar4, counts):
    ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 10,
             str(c), ha='center', va='bottom', color=WHITE, fontsize=9)
style_ax(ax4, 'lncRNA Localization Prediction', 'Compartment', 'Count')
ax4.grid(True, axis='y', color=GRID_CLR, linewidth=0.5)

# P5: Conservation score distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(phylop_scores[expressed],  bins=40, color=ACCENT3, alpha=0.7,
         label='Expressed', edgecolor='none')
ax5.hist(phylop_scores[~expressed], bins=40, color=ACCENT2, alpha=0.7,
         label='Non-expressed', edgecolor='none')
style_ax(ax5, 'Conservation (PhyloP) Scores', 'PhyloP Score', 'Count')
ax5.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=WHITE, framealpha=0.8)

# P6: lncRNA expression distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(mean_expr_lnc, bins=60, color=ACCENT5, alpha=0.8, edgecolor='none')
ax6.axvline(np.median(mean_expr_lnc), color=ACCENT4, linestyle='--',
            linewidth=1.5, label=f'Median={np.median(mean_expr_lnc):.2f}')
style_ax(ax6, 'lncRNA Expression Distribution', 'Mean log2(expr+1)', 'Count')
ax6.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=WHITE, framealpha=0.8)

# P7: Co-expression r distribution
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(r_mat.ravel(), bins=80, color=ACCENT1, alpha=0.8, edgecolor='none')
ax7.axvline( 0.5, color=ACCENT2, linestyle='--', linewidth=1.2, label='|r|=0.5')
ax7.axvline(-0.5, color=ACCENT2, linestyle='--', linewidth=1.2)
style_ax(ax7, 'Co-expression r Distribution', 'Pearson r', 'Count')
ax7.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=WHITE, framealpha=0.8)

# P8: Top 20 hub lncRNAs
ax8 = fig.add_subplot(gs[2, 1])
hub_degrees = lnc_degree_full[top20_hub]
hub_labels  = [f'lnc_{i}' for i in top20_hub]
bar_cols8   = plt.cm.cool(np.linspace(0.2, 0.9, 20))
ax8.barh(range(20), hub_degrees[::-1], color=bar_cols8, edgecolor='none')
ax8.set_yticks(range(20))
ax8.set_yticklabels(hub_labels[::-1], fontsize=6, color=WHITE)
style_ax(ax8, 'Top 20 Hub lncRNAs (connections)', 'Degree', '')
ax8.grid(True, axis='x', color=GRID_CLR, linewidth=0.5)

# P9: Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor(PANEL_BG)
ax9.axis('off')
lines = [
    ("LncRNAEngine Summary",          ACCENT1, 10),
    ("─" * 28,                        ACCENT3,  8),
    (f"Samples:          {N_SAMPLES}", WHITE,    8),
    (f"lncRNAs:          {N_LNCRNA}", WHITE,    8),
    (f"mRNAs:            {N_MRNA}",   WHITE,    8),
    (f"miRNAs:           {N_MIRNA}",  WHITE,    8),
    (f"Co-expr pairs:    {n_coexpr}", WHITE,    8),
    (f"RBPs tested:      {N_RBP}",    WHITE,    8),
    (f"Sig RBPs:         {int(np.sum(rbp_fdr<0.05))}", WHITE, 8),
    (f"Max Jaccard:      {jaccard.max():.4f}", WHITE, 8),
    (f"Nuclear lncRNAs:  {n_nuclear}", WHITE,   8),
    (f"Cytoplasmic:      {n_cytoplasmic}", WHITE, 8),
    (f"PhyloP t-stat:    {t_cons:.3f}", WHITE,  8),
    (f"PhyloP p-value:   {p_cons:.2e}", WHITE,  8),
]
for k, (txt, col, fs) in enumerate(lines):
    ax9.text(0.05, 0.97 - k * 0.065, txt, transform=ax9.transAxes,
             color=col, fontsize=fs, fontfamily='monospace', va='top')

fig.suptitle('LncRNAEngine — Long Non-Coding RNA Regulatory Analysis',
             color=WHITE, fontsize=14, fontweight='bold', y=0.98)

out_path = '/workspace/lncrna_dashboard.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"    Dashboard saved → {out_path}")

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL SUMMARY — LncRNAEngine")
print("=" * 65)
print(f"  Samples analyzed        : {N_SAMPLES}")
print(f"  lncRNAs                 : {N_LNCRNA}")
print(f"  mRNAs                   : {N_MRNA}")
print(f"  miRNAs                  : {N_MIRNA}")
print(f"  Co-expr pairs (sig)     : {n_coexpr}")
print(f"  RBPs tested             : {N_RBP}")
print(f"  Significant RBPs        : {int(np.sum(rbp_fdr<0.05))}")
print(f"  Max Jaccard ceRNA score : {jaccard.max():.4f}")
print(f"  Nuclear lncRNAs         : {n_nuclear} ({100*n_nuclear/N_LNCRNA:.1f}%)")
print(f"  Cytoplasmic lncRNAs     : {n_cytoplasmic} ({100*n_cytoplasmic/N_LNCRNA:.1f}%)")
print(f"  PhyloP t-stat           : {t_cons:.3f}")
print(f"  PhyloP p-value          : {p_cons:.2e}")
print(f"  Max hub degree          : {int(hub_degrees.max())}")
print(f"  Dashboard               : {out_path}")
print("=" * 65)

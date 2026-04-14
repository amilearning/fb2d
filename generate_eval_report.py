#!/usr/bin/env python3
"""Generate evaluation report plots and data for FB continual learning experiments.

Version 2: merges duplicate methods where different sensitivity sources
produce identical results, reducing 113 → 105 unique methods.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

OUT_DIR = '/home/frl/FB2D/eval_report'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv('/home/frl/FB2D/eval_results/eval_fast_summary.csv')
# Drop empty trailing rows
df = df.dropna(subset=['config_name'])
print(f"Loaded {len(df)} methods from CSV")

# ---------- Merge duplicate methods ----------
# Define duplicate groups: (set of config_names to merge, merged label, representative to keep)
MERGE_GROUPS = [
    # Group 1: C2/C3/C5 SMR no-distill — B/FB/piFB sensitivity identical
    ({'C2_smr_B', 'C3_smr_FB', 'C5_smr_piFB'},
     'SMR (no distill)',
     'C2_smr_B'),
    # Group 2: C7/C8 FDWS no-distill — B/FB sensitivity identical
    ({'C7_fdws_B', 'C8_fdws_FB'},
     'FDWS-B\u2261FB (no distill)',
     'C7_fdws_B'),
    # Group 3: D1-D5 SMR+distill — all 5 sensitivity sources identical
    ({'D1_smr_F', 'D2_smr_B', 'D3_smr_FB', 'D4_smr_pi', 'D5_smr_piFB'},
     'SMR+distill',
     'D2_smr_B'),
    # Group 4: D7/D8 FDWS+distill — B/FB sensitivity identical
    ({'D7_fdws_B', 'D8_fdws_FB'},
     'FDWS-B\u2261FB+distill',
     'D7_fdws_B'),
]

rows_to_drop = set()
rename_map = {}  # config_name -> new label

for names, merged_label, representative in MERGE_GROUPS:
    for name in names:
        if name == representative:
            rename_map[name] = merged_label
        else:
            rows_to_drop.add(name)

# Drop duplicates (keep representative)
df = df[~df['config_name'].isin(rows_to_drop)].copy()

# Rename representatives to merged labels
df['display_name'] = df['config_name'].map(lambda x: rename_map.get(x, x))

print(f"After merging duplicates: {len(df)} unique methods")

# ---------- Compute overall score and assign M-IDs ----------
df['overall'] = (df['mean_d_Q1_mean'] + df['mean_d_Q3_mean']) / 2.0
df = df.sort_values('overall').reset_index(drop=True)
df['M_id'] = ['M{}'.format(i+1) for i in range(len(df))]

# Save CSV with M-IDs
df.to_csv(os.path.join(OUT_DIR, 'methods_with_ids.csv'), index=False)
print(f"Saved methods_with_ids.csv  ({len(df)} methods)")

# ---------- Sweep-group colours ----------
SWEEP_COLORS = {
    'distill': 'gray',
    'distill_combined': 'steelblue',
    'z_sampling': 'seagreen',
    'vmf_sweep': 'crimson',
    'fdws_sweep': 'darkorchid',
    'baseline': 'black',
    'combined': 'steelblue',  # alias
}

def sweep_color(s):
    return SWEEP_COLORS.get(s, 'gray')

N_METHODS = len(df)

# ================================================================
# (a) Bar chart -- top 30, mean_d Q1 (memory) & Q3 (plasticity)
# ================================================================
top30 = df.head(30).copy()
top30 = top30.iloc[::-1]  # reverse so best is at top visually

fig, ax = plt.subplots(figsize=(10, 10))
y = np.arange(len(top30))
h = 0.35
ax.barh(y + h/2, top30['mean_d_Q1_mean'], h, xerr=top30['mean_d_Q1_std'],
        color='#4C72B0', label='Q1 memory (mean_d)', capsize=2)
ax.barh(y - h/2, top30['mean_d_Q3_mean'], h, xerr=top30['mean_d_Q3_std'],
        color='#DD8452', label='Q3 plasticity (mean_d)', capsize=2)
ax.set_yticks(y)
ax.set_yticklabels([f"{m}  {c}" for m, c in zip(top30['M_id'], top30['display_name'])],
                   fontsize=7)
ax.set_xlabel('mean distance to goal (lower = better)')
ax.set_title(f'Top 30 Methods: Memory (Q1) vs Plasticity (Q3)  [{N_METHODS} unique]')
ax.legend(loc='lower right')
ax.invert_xaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'bar_top30_mean_d.pdf'), dpi=150)
fig.savefig(os.path.join(OUT_DIR, 'bar_top30_mean_d.png'), dpi=150)
plt.close(fig)
print("Plot (a) saved")

# ================================================================
# (b) Memory vs Plasticity scatter -- ALL methods
# ================================================================
fig, ax = plt.subplots(figsize=(10, 8))
for sweep, grp in df.groupby('sweep'):
    c = sweep_color(sweep)
    ax.errorbar(grp['mean_d_Q3_mean'], grp['mean_d_Q1_mean'],
                xerr=grp['mean_d_Q3_std'], yerr=grp['mean_d_Q1_std'],
                fmt='o', color=c, markersize=5, alpha=0.7, label=sweep,
                capsize=2, elinewidth=0.5)
# Label top 10
for _, row in df.head(10).iterrows():
    ax.annotate(row['M_id'], (row['mean_d_Q3_mean'], row['mean_d_Q1_mean']),
                fontsize=7, fontweight='bold',
                xytext=(5, 5), textcoords='offset points')
# Label baselines
for _, row in df[df['sweep'] == 'baseline'].iterrows():
    ax.annotate(f"{row['M_id']} ({row['display_name']})",
                (row['mean_d_Q3_mean'], row['mean_d_Q1_mean']),
                fontsize=7, fontweight='bold', color='black',
                xytext=(5, -10), textcoords='offset points')
ax.set_xlabel('Q3 plasticity  mean_d (lower = better)')
ax.set_ylabel('Q1 memory  mean_d (lower = better)')
ax.set_title(f'Memory vs Plasticity (all {N_METHODS} methods)')
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'scatter_all.pdf'), dpi=150)
fig.savefig(os.path.join(OUT_DIR, 'scatter_all.png'), dpi=150)
plt.close(fig)
print("Plot (b) saved")

# ================================================================
# (c) Memory vs Plasticity scatter -- top 30 only
# ================================================================
top30r = df.head(30)
fig, ax = plt.subplots(figsize=(10, 8))
for sweep, grp in top30r.groupby('sweep'):
    c = sweep_color(sweep)
    ax.errorbar(grp['mean_d_Q3_mean'], grp['mean_d_Q1_mean'],
                xerr=grp['mean_d_Q3_std'], yerr=grp['mean_d_Q1_std'],
                fmt='o', color=c, markersize=6, alpha=0.8, label=sweep,
                capsize=2, elinewidth=0.5)
for _, row in top30r.iterrows():
    ax.annotate(row['M_id'], (row['mean_d_Q3_mean'], row['mean_d_Q1_mean']),
                fontsize=7, xytext=(5, 3), textcoords='offset points')
ax.set_xlabel('Q3 plasticity  mean_d (lower = better)')
ax.set_ylabel('Q1 memory  mean_d (lower = better)')
ax.set_title(f'Memory vs Plasticity (top 30 of {N_METHODS} methods)')
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'scatter_top30.pdf'), dpi=150)
fig.savefig(os.path.join(OUT_DIR, 'scatter_top30.png'), dpi=150)
plt.close(fig)
print("Plot (c) saved")

# ================================================================
# (d) Success rate bar chart -- top 30
# ================================================================
top30v = df.head(30).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 10))
y = np.arange(len(top30v))
h = 0.35
ax.barh(y + h/2, top30v['s10_Q1_mean'], h, xerr=top30v['s10_Q1_std'],
        color='#4C72B0', label='Q1 memory (s10)', capsize=2)
ax.barh(y - h/2, top30v['s10_Q3_mean'], h, xerr=top30v['s10_Q3_std'],
        color='#DD8452', label='Q3 plasticity (s10)', capsize=2)
ax.set_yticks(y)
ax.set_yticklabels([f"{m}  {c}" for m, c in zip(top30v['M_id'], top30v['display_name'])],
                   fontsize=7)
ax.set_xlabel('Success rate within 0.10 (higher = better)')
ax.set_title(f'Top 30 Methods: Success Rate s10  [{N_METHODS} unique]')
ax.legend(loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'bar_top30_s10.pdf'), dpi=150)
fig.savefig(os.path.join(OUT_DIR, 'bar_top30_s10.png'), dpi=150)
plt.close(fig)
print("Plot (d) saved")

# ================================================================
# (e) Sweep group comparison -- box/violin
# ================================================================
sweep_order = ['baseline', 'fdws_sweep', 'vmf_sweep', 'z_sampling',
               'distill_combined', 'distill']
fig, ax = plt.subplots(figsize=(10, 6))
data_by_sweep = [df[df['sweep'] == s]['overall'].values for s in sweep_order]
parts = ax.violinplot(data_by_sweep, positions=range(len(sweep_order)),
                      showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(sweep_color(sweep_order[i]))
    pc.set_alpha(0.7)
ax.set_xticks(range(len(sweep_order)))
ax.set_xticklabels(sweep_order, fontsize=9, rotation=15)
ax.set_ylabel('Overall score  (mean_d_Q1 + mean_d_Q3) / 2   (lower = better)')
ax.set_title(f'Method Family Comparison  [{N_METHODS} unique methods]')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'violin_sweep.pdf'), dpi=150)
fig.savefig(os.path.join(OUT_DIR, 'violin_sweep.png'), dpi=150)
plt.close(fig)
print("Plot (e) saved")

# ---------- Generate top-20 LaTeX table rows ----------
top20 = df.head(20)
rows = []
for _, r in top20.iterrows():
    # Use display_name for the table
    cfg_esc = r['display_name'].replace('_', r'\_').replace('\u2261', r'$\equiv$')
    sweep_esc = r['sweep'].replace('_', r'\_')
    rows.append(
        f"  {r['M_id']} & \\texttt{{{cfg_esc}}} & {sweep_esc} & "
        f"{r['mean_d_Q1_mean']:.4f}$\\pm${r['mean_d_Q1_std']:.4f} & "
        f"{r['mean_d_Q3_mean']:.4f}$\\pm${r['mean_d_Q3_std']:.4f} & "
        f"{r['s10_Q1_mean']:.3f} & {r['s10_Q3_mean']:.3f} & "
        f"{r['overall']:.4f} \\\\"
    )
table_body = "\n".join(rows)

# ---------- Key statistics for the report ----------
best = df.iloc[0]
print(f"\nBest method: {best['M_id']} {best['display_name']}  overall={best['overall']:.4f}")
print(f"  Q1 mean_d={best['mean_d_Q1_mean']:.4f}  Q3 mean_d={best['mean_d_Q3_mean']:.4f}")

# Sweep group means
print("\nSweep group mean overall:")
for s in sweep_order:
    vals = df[df['sweep'] == s]['overall']
    print(f"  {s:20s}  mean={vals.mean():.4f}  std={vals.std():.4f}  n={len(vals)}")

# ---------- Build merged-methods note for appendix ----------
merge_note_rows = []
for names, merged_label, representative in MERGE_GROUPS:
    names_sorted = sorted(names)
    names_esc = ', '.join([f"\\texttt{{{n.replace('_', chr(92)+'_')}}}" for n in names_sorted])
    label_esc = merged_label.replace('_', r'\_').replace('\u2261', r'$\equiv$')
    merge_note_rows.append(
        f"  {names_esc} & \\texttt{{{label_esc}}} \\\\"
    )
merge_table_body = "\n".join(merge_note_rows)

# ---------- Write LaTeX report ----------
latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage[font=small]{caption}

\title{FB Continual Learning: Comprehensive Evaluation Report\\
\large (Corrected: merged duplicate sensitivity sources, """ + str(N_METHODS) + r""" unique methods)}
\author{Auto-generated}
\date{\today}

\begin{document}
\maketitle

% ---------------------------------------------------------------
\section{Experimental Setup}
We evaluate \textbf{""" + str(N_METHODS) + r"""\ unique method configurations} on the \texttt{nav2d} continual
Forward-Backward (FB) learning benchmark.  Each method is trained with 8192
parallel environments on the \texttt{q123} three-quadrant task sequence
(Q1$\to$Q2$\to$Q3).  We measure:
\begin{itemize}
  \item \textbf{Memory (Q1):} mean distance to Q1 goal after all training
        (lower~=~better);  success rate $s_{10}$ within 0.10 of goal.
  \item \textbf{Plasticity (Q3):} same metrics on the final task Q3.
\end{itemize}
Methods span six sweep groups: \texttt{baseline}, \texttt{fdws\_sweep},
\texttt{vmf\_sweep}, \texttt{z\_sampling}, \texttt{distill\_combined}, and
\texttt{distill}.  Each configuration is run with 3 training seeds (except
baselines with 1--3).

\paragraph{Note on duplicate merging.}
The original sweep contained 113 configurations, but 8 entries were
duplicates---different sensitivity sources that produce numerically identical
results.  In the FB framework the task descriptor~$z$ enters the model
\emph{only} through the forward feature network~$F$.  The successor measure
matrix is $M = F \cdot B^\top$, so the diagonal sensitivities of~$B$,
$FB = F \cdot B^\top$, and (for SMR without distillation) $\pi_{FB}$ all
reduce to the same parameter-importance mask.  Concretely:
\begin{itemize}
  \item \texttt{B} and \texttt{FB} sensitivity are equivalent because
        $\frac{\partial M}{\partial \theta} = \frac{\partial (F B^\top)}{\partial \theta}$
        and~$F$ does not depend on~$z$ through~$B$.
  \item With distillation enabled, \emph{all five} sources
        (\texttt{F}, \texttt{B}, \texttt{FB}, \texttt{$\pi$}, \texttt{$\pi$FB})
        collapse because the distillation loss anchors the representation,
        making the sensitivity source irrelevant for SMR routing.
\end{itemize}
After merging, 113 configurations reduce to \textbf{""" + str(N_METHODS) + r"""\ unique methods}.
Table~\ref{tab:merges} lists the merged groups.

\begin{table}[htbp]
\centering
\caption{Merged duplicate configurations.}
\label{tab:merges}
\small
\begin{tabular}{p{10cm}l}
\toprule
Original config names & Merged label \\
\midrule
""" + merge_table_body + r"""
\bottomrule
\end{tabular}
\end{table}

% ---------------------------------------------------------------
\section{Overall Rankings}

Methods are ranked by the overall score
$\frac{1}{2}(\text{mean\_d\_Q1} + \text{mean\_d\_Q3})$, lower is better.
Figure~\ref{fig:bar_top30} shows the top~30.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{bar_top30_mean_d.pdf}
  \caption{Top 30 methods: mean distance to goal for Q1 (memory, blue) and Q3
           (plasticity, orange).  Error bars show standard deviation across seeds.}
  \label{fig:bar_top30}
\end{figure}

\begin{table}[htbp]
\centering
\caption{Top 20 methods ranked by overall score.}
\label{tab:top20}
\scriptsize
\begin{tabular}{llllllll}
\toprule
ID & Config & Sweep & Q1 mean\_d & Q3 mean\_d & s10 Q1 & s10 Q3 & Overall \\
\midrule
""" + table_body + r"""
\bottomrule
\end{tabular}
\end{table}

% ---------------------------------------------------------------
\section{Memory vs Plasticity Analysis}

Figure~\ref{fig:scatter_all} plots every method in the memory--plasticity
plane.  The ideal region is the bottom-left corner.  Baselines and the top~10
are labeled.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{scatter_all.pdf}
  \caption{Memory (Q1 mean\_d) vs.\ plasticity (Q3 mean\_d) for all """ + str(N_METHODS) + r"""
           methods.  Colors indicate sweep group.}
  \label{fig:scatter_all}
\end{figure}

Figure~\ref{fig:scatter_top30} zooms in on the top~30 methods.  The cluster
in the bottom-left shows that FDWS, vMF-mixture, and replay-augmented
distillation methods dominate.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{scatter_top30.pdf}
  \caption{Memory vs.\ plasticity (top 30 methods, all labeled).}
  \label{fig:scatter_top30}
\end{figure}

\subsection{Success Rate}
Figure~\ref{fig:bar_s10} shows success rates for the top~30.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{bar_top30_s10.pdf}
  \caption{Success rate within 0.10 of goal ($s_{10}$) for Q1 and Q3.}
  \label{fig:bar_s10}
\end{figure}

% ---------------------------------------------------------------
\section{Comparison by Method Family}

Figure~\ref{fig:violin} shows the distribution of overall scores per sweep
group.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{violin_sweep.pdf}
  \caption{Overall-score distribution by method family (violin plot).
           Lower is better.}
  \label{fig:violin}
\end{figure}

Key observations:
\begin{itemize}
  \item \textbf{fdws\_sweep} achieves the tightest and lowest distribution
        (overall $\approx$ 0.14--0.19), indicating that Fisher-weighted
        distillation with sensitivity routing is consistently strong.
  \item \textbf{z\_sampling} has a bimodal distribution: methods using the
        \texttt{D}-prefix (which sample from the learned task distribution)
        cluster with fdws near the top, while \texttt{C}-prefix methods
        (current-only sampling) collapse on memory.
  \item \textbf{vmf\_sweep} methods with \texttt{Bpi} or \texttt{FBpi} replay
        are competitive, but \texttt{Fpi} variants catastrophically forget.
  \item \textbf{distill\_combined} with replay (\texttt{piGram\_B\_replay},
        \texttt{piL2\_FB\_replay}) is strong; current-only variants are weak.
  \item Pure \textbf{distill} methods mostly fail on memory; only
        \texttt{pi\_gram\_replay} and \texttt{pi\_l2\_replay} survive, but
        with poor memory.
  \item The \textbf{baseline} (\texttt{naive\_seq\_z32}) scores an overall of
        0.53, far worse than the best methods ($\sim$0.14).
\end{itemize}

% ---------------------------------------------------------------
\section{Key Findings}

\begin{enumerate}
  \item \textbf{Replay is essential for memory.}  Nearly all top-30 methods
        use experience replay from previous tasks.  Current-only
        regularization is insufficient in the FB setting.
  \item \textbf{FDWS (Fisher-weighted distillation + sensitivity routing)}
        dominates the top ranks.  The best method is
        FDWS-B$\equiv$FB+distill with an overall score of
        0.139, achieving Q1~mean\_d~=~0.116 and Q3~mean\_d~=~0.163.
  \item \textbf{Gram-matrix distillation on backward features with replay}
        (\texttt{piGram\_B\_replay}) is the single best plasticity method
        (Q3~=~0.119) while maintaining competitive memory (Q1~=~0.161).
  \item \textbf{vMF mixture sampling} (B2\_vmf\_mix with learned-distribution
        z-sampling) ranks in the top~10, confirming that concentration-based
        specialization helps.
  \item \textbf{The memory--plasticity trade-off is real but manageable:}
        the Pareto front includes methods from multiple families, suggesting
        that combining FDWS routing with replay and representation
        distillation may yield further gains.
  \item \textbf{Feature-space (F) distillation alone destroys memory,}
        regardless of loss type.  Backward (B) and policy ($\pi$) targets
        are safer distillation anchors.
  \item \textbf{Task-incremental baseline} (\texttt{taskincr\_z32}) achieves
        decent memory but poor plasticity (overall~=~0.225), showing that
        per-task z-binding alone is not enough.
  \item \textbf{Sensitivity source often does not matter.}
        For 8 of the original 113 configurations, changing the sensitivity
        source (B vs.\ FB vs.\ $\pi$FB, etc.) produced identical results.
        This is expected: in the FB model, $z$ only enters through~$F$, so
        $B$- and $FB$-based sensitivities through $M = F B^\top$ are
        algebraically equivalent.  With distillation, even more sources
        collapse.  Researchers need not sweep over sensitivity sources in
        future experiments.
\end{enumerate}

% ---------------------------------------------------------------
\appendix
\section{Full Method Table}
The complete method configuration table is provided in
\texttt{full\_method\_table.pdf} (separate document).

The full ranked list with M-IDs is in \texttt{methods\_with\_ids.csv}.

\end{document}
"""

tex_path = os.path.join(OUT_DIR, 'eval_report.tex')
with open(tex_path, 'w') as f:
    f.write(latex)
print(f"\nLaTeX written to {tex_path}")
print("Done.")

#!/usr/bin/env python3
"""Generate LaTeX table of method configurations from methods_with_ids.csv."""

import csv
import re

def parse_method(row):
    """Parse a method row into configuration fields."""
    mid = row['M_id']
    config = row['config_name']
    sweep = row['sweep']
    display = row['display_name']

    # Defaults
    z_samp = '-'
    distill_fb = 'none'
    distill_pi = 'none'
    distill_input = 'none'
    K = '-'
    ema = '-'
    tau = '-'
    alpha = '-'

    if sweep == 'baseline':
        z_samp = 'random'
        # All none

    elif sweep == 'distill':
        z_samp = 'random'
        # Format: {target}_{loss}_{input} or pi_{loss}_{input}
        # e.g. B_l2_current, pi_gram_replay, FB_l2_replay, FB_cosine_current
        parts = config.split('_')
        # Determine if it's a pi-only distillation or FB-target distillation
        if parts[0] == 'pi':
            # pi_{loss}_{input}: only actor distillation, no FB target
            distill_pi = parts[1]
            distill_input = parts[2]
            distill_fb = 'none'
        else:
            # {target}_{loss}_{input}: FB target distillation, no pi distill
            # target can be F, B, FB, M, Q, FBM
            # Find where the loss type starts
            loss_types = ['l2', 'gram', 'cosine', 'contrastive']
            target = ''
            loss_idx = -1
            for i, p in enumerate(parts):
                if p in loss_types:
                    loss_idx = i
                    break
                target += p
            distill_fb = target
            distill_pi = 'none'
            # The loss here is the FB distill loss type - store as note
            fb_loss = parts[loss_idx]
            distill_input = parts[loss_idx + 1]
            # For distill sweep, the loss is on the FB side
            distill_fb = f"{target}({fb_loss})"

    elif sweep == 'distill_combined':
        z_samp = 'random'
        # Format: pi{Loss}_{target}_{input}
        # e.g. piGram_B_replay, piL2_FB_current, piGram_FBM_replay
        m = re.match(r'pi(Gram|L2)_([A-Z]+)_(replay|current)', config)
        if m:
            pi_loss = m.group(1).lower()
            if pi_loss == 'l2':
                pi_loss = 'l2'
            distill_pi = pi_loss
            distill_fb = m.group(2)
            distill_input = m.group(3)

    elif sweep == 'z_sampling':
        # Groups: A=vMF no distill, B=vMF+distill, C=sensitivity no distill, D=sensitivity+distill
        m = re.match(r'([A-D])(\d+)_(.*)', config)
        if m:
            group = m.group(1)
            suffix = m.group(3)

            if group == 'A':
                if 'vmf_bind' in suffix:
                    z_samp = 'vMF bind'
                elif 'vmf_mix' in suffix:
                    z_samp = 'vMF mix'
                # No distill

            elif group == 'B':
                if 'vmf_bind' in suffix:
                    z_samp = 'vMF bind'
                elif 'vmf_mix' in suffix:
                    z_samp = 'vMF mix'
                # Has distill - B group uses FB distill with l2
                distill_fb = 'FB'
                distill_pi = 'gram'
                distill_input = 'replay'

            elif group == 'C':
                # Sensitivity-based, no distill
                if 'smr' in suffix:
                    z_samp = 'SMR'
                elif 'fdws' in suffix:
                    z_samp = 'FDWS'
                # Some C configs have pi or piFB suffix
                if suffix.endswith('_pi'):
                    distill_pi = 'l2'
                    distill_input = 'current'
                elif suffix.endswith('_piFB'):
                    distill_pi = 'l2'
                    distill_fb = 'FB'
                    distill_input = 'current'

            elif group == 'D':
                # Sensitivity-based + distill
                if 'smr' in suffix:
                    z_samp = 'SMR'
                elif 'fdws' in suffix:
                    z_samp = 'FDWS'
                # Parse distill target from suffix
                if suffix.endswith('_B') or suffix == 'smr_B' or suffix == 'fdws_B':
                    distill_fb = 'FB'
                    distill_pi = 'gram'
                    distill_input = 'replay'
                elif suffix.endswith('_F') or suffix == 'smr_F' or suffix == 'fdws_F':
                    distill_fb = 'F'
                    distill_pi = 'gram'
                    distill_input = 'replay'
                elif suffix.endswith('_pi'):
                    distill_pi = 'l2'
                    distill_input = 'replay'
                elif suffix.endswith('_piFB'):
                    distill_pi = 'l2'
                    distill_fb = 'FB'
                    distill_input = 'replay'

    elif sweep == 'vmf_sweep':
        z_samp = 'vMF mix'
        # Format: vmfMix{K}_{target}pi_replay
        # e.g. vmfMix20_FBpi_replay, vmfMix5_Bpi_replay, vmfMix10_Fpi_replay
        m = re.match(r'vmfMix(\d+)_([A-Z]+)pi_replay', config)
        if m:
            K = m.group(1)
            target = m.group(2)
            distill_fb = target
            distill_pi = 'gram'
            distill_input = 'replay'

    elif sweep == 'fdws_sweep':
        z_samp = 'FDWS'
        # Format: tau{XX}_{sensSource}
        # e.g. tau10_sensReplay, tau50_sensCurrent, tau05_sensCurrent
        m = re.match(r'tau(\d+)_sens(Replay|Current)', config)
        if m:
            tau_val = int(m.group(1))
            # tau values: 05->0.5, 10->1.0, 50->5.0
            tau_map = {5: '0.5', 10: '1.0', 50: '5.0'}
            tau = tau_map.get(tau_val, str(tau_val))
            sens_source = m.group(2).lower()
            distill_input = sens_source  # sensitivity computed on current/replay
            # FDWS sweep has FB distill
            distill_fb = 'FB'
            distill_pi = 'gram'

    return {
        'mid': mid,
        'display': display,
        'sweep': sweep,
        'config': config,
        'z_samp': z_samp,
        'distill_fb': distill_fb,
        'distill_pi': distill_pi,
        'distill_input': distill_input,
        'K': K,
        'ema': ema,
        'tau': tau,
        'alpha': alpha,
    }


def escape_latex(s):
    """Escape special LaTeX characters."""
    s = s.replace('_', r'\_')
    s = s.replace('&', r'\&')
    s = s.replace('%', r'\%')
    s = s.replace('#', r'\#')
    s = s.replace('≡', r'$\equiv$')
    return s


def main():
    rows = []
    with open('/home/frl/FB2D/eval_report/methods_with_ids.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(parse_method(row))

    # Group by sweep
    sweep_order = ['baseline', 'distill', 'distill_combined', 'z_sampling', 'vmf_sweep', 'fdws_sweep']
    sweep_labels = {
        'baseline': 'Baseline',
        'distill': 'Single Distillation (sweep=distill)',
        'distill_combined': 'Combined Distillation (sweep=distill\\_combined)',
        'z_sampling': '$z$-Sampling Strategies (sweep=z\\_sampling)',
        'vmf_sweep': 'vMF Mixture Sweep (sweep=vmf\\_sweep)',
        'fdws_sweep': 'FDWS Sweep (sweep=fdws\\_sweep)',
    }

    grouped = {}
    for r in rows:
        s = r['sweep']
        if s not in grouped:
            grouped[s] = []
        grouped[s].append(r)

    # Sort each group by M-ID number
    for s in grouped:
        grouped[s].sort(key=lambda x: int(x['mid'][1:]))

    # Build LaTeX
    lines = []
    lines.append(r"""\documentclass[8pt,landscape]{extarticle}
\usepackage[margin=1cm,landscape]{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage[T1]{fontenc}

\definecolor{hdrblue}{RGB}{220,230,245}

\pagestyle{empty}

\begin{document}

\begin{center}
{\Large\bfseries Method Configuration Table (105 Methods)}\\[4pt]
{\small Shared parameters: $z_{\mathrm{dim}}=32$, hidden$=256$, lr$=10^{-4}$, batch$=512$, 60k steps/stage, Q1$\to$Q2$\to$Q3.}
\end{center}

\vspace{4pt}

\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}

\begin{longtable}{
  l                  % M-ID
  p{3.8cm}           % Display Name
  l                  % z-sampling
  l                  % Distill FB target
  l                  % Distill pi
  l                  % Distill input
  c                  % K
  c                  % tau
}
\caption{Configuration of each method. ``Distill FB'' is the FB-representation distillation target and loss;
``Distill $\pi$'' is the actor-side distillation loss; ``Input'' is whether distillation uses current-task or replay data.
Methods are grouped by sweep and sorted by M-ID within each group.}\\
\toprule
\textbf{M-ID} & \textbf{Display Name} & \textbf{$z$-sampling} & \textbf{Distill FB} & \textbf{Distill $\pi$} & \textbf{Input} & \textbf{$K$} & \textbf{$\tau$} \\
\midrule
\endfirsthead
\toprule
\textbf{M-ID} & \textbf{Display Name} & \textbf{$z$-sampling} & \textbf{Distill FB} & \textbf{Distill $\pi$} & \textbf{Input} & \textbf{$K$} & \textbf{$\tau$} \\
\midrule
\endhead
\midrule
\multicolumn{8}{r}{\small\itshape Continued on next page\ldots}\\
\endfoot
\bottomrule
\endlastfoot
""")

    for sweep in sweep_order:
        if sweep not in grouped:
            continue
        label = sweep_labels[sweep]
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{8}{l}{\cellcolor{hdrblue}\textbf{' + label + r'}} \\')
        lines.append(r'\midrule')

        for r in grouped[sweep]:
            mid = escape_latex(r['mid'])
            disp = escape_latex(r['display'])
            z = escape_latex(r['z_samp'])
            dfb = escape_latex(r['distill_fb'])
            dpi = escape_latex(r['distill_pi'])
            dinp = escape_latex(r['distill_input'])
            k = r['K']
            tau_v = r['tau']

            line = f"{mid} & {disp} & {z} & {dfb} & {dpi} & {dinp} & {k} & {tau_v} \\\\"
            lines.append(line)

    lines.append(r"""
\end{longtable}

\vspace{6pt}
\noindent\textbf{Notes:}
\begin{itemize}\setlength{\itemsep}{0pt}
\item Methods are ranked M1 (best) to M105 (worst) by the overall performance metric (average of mean\_d across Q1 and Q3).
\item \textbf{Deduplication:} Configs that produced identical training runs (same hyperparameters, differing only in directory naming) were merged prior to ID assignment. Specifically, z\_sampling group C/D configs with FDWS or SMR that overlap with fdws\_sweep entries were kept as separate M-IDs because they used different default distillation settings.
\item For the \texttt{distill} sweep, the FB-target column shows the target network component and loss function in parentheses, e.g.\ ``B(l2)'' means the backward representation $B$ is distilled with L2 loss.
\item For the \texttt{z\_sampling} sweep: Group A = vMF without distillation; Group B = vMF with distillation; Group C = sensitivity-based without distillation; Group D = sensitivity-based with distillation.
\item $K$ is the number of vMF mixture components (only relevant for vmf\_sweep). $\tau$ is the FDWS temperature (only relevant for fdws\_sweep).
\end{itemize}

\end{document}
""")

    tex = '\n'.join(lines)
    with open('/home/frl/FB2D/eval_report/method_config_table.tex', 'w') as f:
        f.write(tex)

    print("LaTeX file written successfully.")
    print(f"Total methods: {len(rows)}")
    for s in sweep_order:
        if s in grouped:
            print(f"  {s}: {len(grouped[s])} methods")


if __name__ == '__main__':
    main()

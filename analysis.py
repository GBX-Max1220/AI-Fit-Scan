#!/usr/bin/env python3
"""
AI Fit Scan — Full Analysis Pipeline
1. AI-Washing Core Findings Figure
2. Data Cleaning (description column)
3. AI Buzzword Analysis (F vs T)
"""

import csv
import re
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ──────────────────────────────────────────────────────────
# 0. Load data
# ──────────────────────────────────────────────────────────
CSV_PATH = Path("/mnt/c/Users/gbx12/Desktop/AI-Fit-Scan-repo/ai_fit_scan_full.csv")
OUT_DIR  = Path("/mnt/c/Users/gbx12/Desktop/AI-Fit-Scan-repo")
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Filter to T/Q/F only
tqf = [r for r in rows if r['l2_label'] in ('T', 'Q', 'F')]
t_apps = [r for r in rows if r['l2_label'] == 'T']
q_apps = [r for r in rows if r['l2_label'] == 'Q']
f_apps = [r for r in rows if r['l2_label'] == 'F']

print(f"T (True AI): {len(t_apps)}   Q (Questionable): {len(q_apps)}   F (Fake AI-Washing): {len(f_apps)}")

# ──────────────────────────────────────────────────────────
# 1. AI-Washing Core Findings Figure
# ──────────────────────────────────────────────────────────

# Representative apps per category (sorted by rating_count descending for recognizability)
def sort_by_popularity(apps, top_n=4):
    """Pick top N apps by rating_count (descending)."""
    valid = [a for a in apps if a['rating_count'] and a['rating_count'].strip()]
    valid.sort(key=lambda x: float(x['rating_count']), reverse=True)
    return valid[:top_n]

T_REPS = sort_by_popularity(t_apps, 4)
Q_REPS = sort_by_popularity(q_apps, 4)
F_REPS = sort_by_popularity(f_apps, 4)

print("\nT reps:", [a['name'] for a in T_REPS])
print("Q reps:", [a['name'] for a in Q_REPS])
print("F reps:", [a['name'] for a in F_REPS])

# Layout: 3 columns (T, Q, F)
categories = [
    ("True AI", "T", T_REPS, "#2E86AB"),      # professional teal
    ("Questionable", "Q", Q_REPS, "#E09F3E"),  # amber/gold
    ("AI-Washing", "F", F_REPS, "#A23B3B"),    # brick red
]

# Also compute stats for annotations
def avg_rating(apps):
    vals = [float(a['rating']) for a in apps if a['rating'] and a['rating'].strip()]
    return round(sum(vals)/len(vals), 2) if vals else 0

def avg_installs(apps):
    vals = []
    for a in apps:
        v = a['installs'].strip().replace(',', '').replace('+', '')
        if v and v.isdigit():
            vals.append(int(v))
    return round(sum(vals)/len(vals)) if vals else 0

stats = []
for cat_label, cat_key, apps, color in categories:
    ar = avg_rating(apps)
    ai = avg_installs(apps)
    stats.append((cat_label, cat_key, apps, color, ar, ai))

# ── Build the figure ──
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.patch.set_facecolor('#FAFAFA')

ax.set_xlim(0, 30)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(15, 9.6, '"AI-Washing" in Fitness Apps',
        fontsize=22, fontweight='bold', ha='center', va='top',
        fontfamily='sans-serif', color='#222222')
ax.text(15, 9.15, 'Labeling gap between marketing claims and actual AI functionality',
        fontsize=12, ha='center', va='top', fontfamily='sans-serif',
        color='#666666', fontstyle='italic')

# Column positions
col_centers = [5.5, 15, 24.5]
col_widths = [7.0, 7.0, 7.0]

for idx, (cat_label, cat_key, reps_col, color, ar, ai_avg) in enumerate(stats):
    cx = col_centers[idx]
    cw = col_widths[idx]
    x0 = cx - cw/2
    y_top = 8.5  # top of header area
    n_apps = len(t_apps) if cat_key == 'T' else (len(q_apps) if cat_key == 'Q' else len(f_apps))

    # ── Column header badge ──
    badge = FancyBboxPatch((x0 + 0.3, y_top - 0.5), cw - 0.6, 0.9,
                           boxstyle="round,pad=0.08", facecolor=color, edgecolor='none', alpha=0.85)
    ax.add_patch(badge)
    ax.text(cx, y_top - 0.05, f'{cat_label}',
            fontsize=14, fontweight='bold', ha='center', va='center',
            color='white', fontfamily='sans-serif')

    # ── Count badge ──
    ax.text(cx, y_top - 0.95, f'n = {n_apps}',
            fontsize=13, ha='center', va='top', color=color,
            fontweight='bold', fontfamily='sans-serif')

    # ── Key stat line ──
    ax.text(cx, y_top - 1.45, f'Avg Rating: {ar} ★   |   Avg Installs: {ai_avg:,}',
            fontsize=9, ha='center', va='top', color='#555555',
            fontfamily='sans-serif')

    # ── App cards ──
    card_y_start = y_top - 2.1
    card_h = 0.55
    card_gap = 0.18

    for j, app in enumerate(reps_col):
        cy = card_y_start - j * (card_h + card_gap)

        # card background
        card = FancyBboxPatch((x0 + 0.5, cy), cw - 1.0, card_h,
                               boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor='#DDDDDD', linewidth=0.6)
        ax.add_patch(card)

        # tiny indicator
        ax.plot([x0 + 0.5, x0 + 0.5], [cy, cy + card_h],
                color=color, linewidth=2.5, solid_capstyle='round',
                transform=ax.transData)

        # app name (bold)
        name = app['name']
        if len(name) > 20:
            name = name[:18] + '…'
        ax.text(cx, cy + card_h/2 + 0.06, name,
                fontsize=9.5, fontweight='bold', ha='center', va='center',
                color='#222222', fontfamily='sans-serif')

        # sub info
        rating_stars = f"{float(app['rating']):.1f}★" if app['rating'] else 'N/A'
        installs_str = app['installs'].replace('+', '') if app['installs'] else '?'
        ax.text(cx, cy + card_h/2 - 0.18, f'{rating_stars}  |  {installs_str}',
                fontsize=7.5, ha='center', va='center', color='#888888',
                fontfamily='sans-serif')

    # ── Bottom insight ──
    if cat_key == 'T':
        insight = '✓ Genuine ML/AI features\n(pose detection, LLM coaching, planning)'
        insight_color = '#2E86AB'
    elif cat_key == 'Q':
        insight = '⚠ Inflated AI claims\n(minimal verifiable AI functionality)'
        insight_color = '#E09F3E'
    else:
        insight = '✗ AI-Washing: AI in name only\nno substantive AI functionality'
        insight_color = '#A23B3B'

    ax.text(cx, card_y_start - len(reps_col) * (card_h + card_gap) - 0.3,
            insight, fontsize=9, ha='center', va='top',
            color=insight_color, fontfamily='sans-serif',
            fontweight='bold')

# ── Footer / Methodology ──
ax.text(15, 0.25,
        'Methodology: Apps were manually labeled by two annotators as True AI (T), Questionable (Q), or AI-Washing (F)\n'
        'based on actual AI functionality described in the app listing vs. marketing language. Source: Google Play Store (2024-2025).',
        fontsize=7.5, ha='center', va='bottom', color='#999999',
        fontfamily='sans-serif')

# ── Legend: color bar ──
legend_elements = [
    mpatches.Patch(facecolor='#2E86AB', edgecolor='none', alpha=0.85, label='True AI (T) — 35 apps'),
    mpatches.Patch(facecolor='#E09F3E', edgecolor='none', alpha=0.85, label='Questionable (Q) — 4 apps'),
    mpatches.Patch(facecolor='#A23B3B', edgecolor='none', alpha=0.85, label='AI-Washing (F) — 22 apps'),
]
legend = ax.legend(handles=legend_elements, loc='lower center',
                   fontsize=9, ncol=3, framealpha=0.7,
                   bbox_to_anchor=(0.5, -0.04))
legend.get_frame().set_facecolor('#FAFAFA')

plt.tight_layout()
fig_path = OUT_DIR / 'ai_washing_figure.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n✅ Figure saved: {fig_path}")


# ──────────────────────────────────────────────────────────
# 2. Data Cleaning: Clean description column
# ──────────────────────────────────────────────────────────

def clean_description(text):
    if not text:
        return ''
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excess newlines (replace 2+ with a single space)
    text = re.sub(r'\n\s*\n+', ' ', text)
    # Remove carriage returns
    text = text.replace('\r', '')
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    # Replace special Unicode bullets/symbols with simpler markers or strip
    # Remove emoji and special symbols (keep basic punctuation and letters)
    # Keep a-zA-Z0-9 . , ! ? ' " ( ) - : ; @ # $ % & + / = and basic whitespace
    text = re.sub(r'[^\x20-\x7E\xA0-\xFF\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text)
    # Collapse spaces again
    text = re.sub(r' +', ' ', text)
    return text.strip()

# Test a couple
print("\n=== Data Cleaning ===")
print(f"Before: {repr(rows[0]['description'][:100])}")
print(f"After:  {repr(clean_description(rows[0]['description'][:200]))}")

cleaned_rows = []
for r in rows:
    new_r = dict(r)
    new_r['description'] = clean_description(r['description'])
    cleaned_rows.append(new_r)

clean_csv_path = OUT_DIR / 'ai_fit_scan_full_clean.csv'
with open(clean_csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(cleaned_rows)
print(f"✅ Clean CSV saved: {clean_csv_path} ({len(cleaned_rows)} rows)")


# ──────────────────────────────────────────────────────────
# 3. AI Buzzword Analysis: F vs T description language
# ──────────────────────────────────────────────────────────

print("\n=== AI Buzzword Analysis ===")

# Define AI-related keywords to look for
ai_keywords = [
    'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural',
    'smart', 'intelligent', 'personalized', 'adaptive', 'smart', 'algorithm',
    'powered by ai', 'ai-powered', 'ai driven', 'ai coach', 'ai trainer',
    'virtual coach', 'virtual trainer', 'smart coach',
    'computer vision', 'pose estimation', 'pose detection',
    'natural language', 'chatbot', 'gpt', 'llm', 'large language model',
    'recommendation', 'predictive', 'automated', 'real-time',
    'analytics', 'insight', 'optimize', 'tracking',
]

def extract_words(text):
    """Lower-case and tokenize."""
    text = text.lower()
    # Remove punctuation, keep words
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return [w for w in words if len(w) > 1]

def count_ai_terms(apps, keywords):
    """Count how many apps use each keyword in their description."""
    kw_counts = Counter()
    for app in apps:
        desc = app['description'].lower()
        for kw in keywords:
            if kw in desc:
                kw_counts[kw] += 1
    return kw_counts

# Count for F and T
f_kw_counts = count_ai_terms(f_apps, ai_keywords)
t_kw_counts = count_ai_terms(t_apps, ai_keywords)

# Normalize by group size
n_f = len(f_apps)
n_t = len(t_apps)

print(f"\n--- Top AI terms in F (AI-Washing) apps (n={n_f}) ---")
for kw, cnt in f_kw_counts.most_common(20):
    pct = cnt / n_f * 100
    # Also show T percentage for comparison
    t_cnt = t_kw_counts.get(kw, 0)
    t_pct = t_cnt / n_t * 100
    ratio = f"{pct/t_pct:.1f}x" if t_pct > 0 else "∞ (0 in T)"
    bar = '█' * int(pct / 5) + '░' * max(0, 20 - int(pct / 5))
    print(f"  {kw:25s}  {bar}  {cnt:3d}/{n_f:2d} ({pct:5.1f}%)   vs T: {t_cnt:3d}/{n_t:2d} ({t_pct:5.1f}%)  [{ratio}]")

print(f"\n--- Top AI terms in T (True AI) apps (n={n_t}) ---")
for kw, cnt in t_kw_counts.most_common(20):
    pct = cnt / n_t * 100
    f_cnt = f_kw_counts.get(kw, 0)
    f_pct = f_cnt / n_f * 100
    ratio = f"{pct/max(f_pct,0.01):.1f}x"
    bar = '█' * int(pct / 5) + '░' * max(0, 20 - int(pct / 5))
    print(f"  {kw:25s}  {bar}  {cnt:3d}/{n_t:2d} ({pct:5.1f}%)   vs F: {f_cnt:3d}/{n_f:2d} ({f_pct:5.1f}%)  [{ratio}]")

# Also find the most distinctive words (biggest gap)
print("\n--- Most disproportionately used in F vs T ---")
f_ratio = {}
for kw in ai_keywords:
    f_pct = f_kw_counts.get(kw, 0) / max(n_f, 1) * 100
    t_pct = t_kw_counts.get(kw, 0) / max(n_t, 1) * 100
    if f_pct > 0 and t_pct == 0:
        f_ratio[kw] = float('inf')
    elif f_pct > 0:
        f_ratio[kw] = f_pct / max(t_pct, 0.01)
        
for kw, ratio in sorted(f_ratio.items(), key=lambda x: -x[1] if x[1] != float('inf') else 999)[:15]:
    f_cnt = f_kw_counts.get(kw, 0)
    t_cnt = t_kw_counts.get(kw, 0)
    ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "∞"
    print(f"  {kw:25s}  F:{f_cnt:3d}/{n_f}  T:{t_cnt:3d}/{n_t}  ratio={ratio_str}")

# ── Buzzword comparison figure ──
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig2.patch.set_facecolor('#FAFAFA')

# Top 10 most common in F
f_top10 = f_kw_counts.most_common(10)
t_top10 = t_kw_counts.most_common(10)

kw_names = [kw for kw, _ in f_top10]
f_vals = [cnt/n_f*100 for _, cnt in f_top10]
t_vals = [t_kw_counts.get(kw, 0)/n_t*100 for kw, _ in f_top10]

x = np.arange(len(kw_names))
w = 0.35

ax1.bar(x - w/2, f_vals, w, label='F (AI-Washing)', color='#A23B3B', alpha=0.85)
ax1.bar(x + w/2, t_vals, w, label='T (True AI)', color='#2E86AB', alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(kw_names, rotation=35, ha='right', fontsize=9)
ax1.set_ylabel('% of apps containing term', fontsize=11)
ax1.set_title('Top AI Buzzwords: AI-Washing (F) vs True AI (T)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_ylim(0, max(max(f_vals), max(t_vals)) * 1.2)
ax1.grid(axis='y', alpha=0.3)

# Top T terms 
kw_names_t = [kw for kw, _ in t_top10]
t_vals_t = [cnt/n_t*100 for _, cnt in t_top10]
f_vals_t = [f_kw_counts.get(kw, 0)/n_f*100 for kw, _ in t_top10]

x2 = np.arange(len(kw_names_t))
ax2.bar(x2 - w/2, t_vals_t, w, label='T (True AI)', color='#2E86AB', alpha=0.85)
ax2.bar(x2 + w/2, f_vals_t, w, label='F (AI-Washing)', color='#A23B3B', alpha=0.85)
ax2.set_xticks(x2)
ax2.set_xticklabels(kw_names_t, rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('% of apps containing term', fontsize=11)
ax2.set_title('Top AI Buzzwords: True AI (T) vs AI-Washing (F)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_ylim(0, max(max(t_vals_t), max(f_vals_t)) * 1.2)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
buzz_path = OUT_DIR / 'ai_buzzword_comparison.png'
plt.savefig(buzz_path, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print(f"\n✅ Buzzword figure saved: {buzz_path}")

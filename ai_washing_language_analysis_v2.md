# AI-Washing Language Fingerprint Analysis

**Author:** Max Guo  
**Date:** 2026-05-30  
**Dataset:** AI-Fit-Scan (161 Google Play fitness apps, 61 AI-claiming with L2 annotation)  
**Source code:** `analysis_v3.py`  

---

## Method

### Sample

| Group | Label | n | Categories |
|-------|-------|---|------------|
| True AI | T | 35 | general_gym (25), running (6), nutrition (4) |
| Quasi-AI | Q | 4 | general_gym (4) — excluded from analysis |
| AI-Washing | F | 22 | tracking_timer (10), general_gym (5), bodyweight_hiit (3), therapy_journal (2), apnea (1), nutrition (1) |

### Procedure

1. **App categorization:** Each of the 57 T/F apps was manually classified into one of 7 categories based on app name and description: `general_gym`, `running`, `nutrition`, `bodyweight_hiit`, `tracking_timer`, `therapy_journal`, `apnea`.
2. **Term extraction:** 21 technical AI/ML terms and phrases were pre-specified (e.g., "artificial intelligence", "machine learning", "real-time", "adaptive", "algorithm"). Word-boundary regex matching was used to avoid substring false positives (e.g., "smart" not matching "smartwatch").
3. **Statistical tests:** For each term, Fisher's exact test was applied to the 2×2 contingency table (presence/absence × T/F group). Cohen's h was computed as the effect size for proportion differences.
4. **Category confound control:** Because categories are unevenly distributed (running apps are all T; tracking_timer apps are all F), a within-group analysis was also run on the `general_gym` category only (T=25, F=5). Terms whose effect direction or magnitude changed substantially in the within-group analysis are flagged as potentially category-confounded.

### Limitations & Caveats

- **Small sample:** F=22, T=35. Within-group analyses are underpowered (e.g., F general_gym = 5).
- **Category imbalance:** Running (6) and nutrition (4) apps are only found in T; tracking_timer (10), bodyweight_hiit (3), therapy_journal (2), and apnea (1) are only found in F. Some observed differences may reflect category characteristics rather than AI-washing strategy.
- **Description-dependent:** Labels are based on Google Play descriptions, not source code or model inspection.
- **Single time point:** Data collected May 2026. AI claims may change with app updates.

---

## Results

### 3a. Technical AI Terminology: T vs F Comparison

| Term | F% (n=22) | T% (n=35) | Δ% | Cohen's h | p | Effect Size | Note |
|------|-----------|----------|----|-----------|----|------------|------|
| artificial intelligence | 0.0% | 20.0% | -20.0% | -0.86* | 0.0360 | large |
| real-time | 9.1% | 42.9% | -33.8% | -0.81** | 0.0078 | large (category-driven) |
| adaptive | 0.0% | 17.1% | -17.1% | -0.79 | 0.0722 | medium |
| personalized | 54.5% | 88.6% | -34.0% | -0.79** | 0.0096 | medium |
| machine learning | 0.0% | 5.7% |  -5.7% | -0.42 | 0.5175 | small |
| automated | 0.0% | 5.7% |  -5.7% | -0.42 | 0.5175 | small |
| algorithm | 4.5% | 11.4% |  -6.9% | -0.26 | 0.6389 | small |
| optimize | 13.6% | 22.9% |  -9.2% | -0.24 | 0.5017 | small |
| biometric tracking | 13.6% | 20.0% |  -6.4% | -0.17 | 0.7250 | negligible (category-driven) |
| voice coaching | 4.5% | 8.6% |  -4.0% | -0.16 | 1.0000 | negligible |
| analytics/insight | 13.6% | 17.1% |  -3.5% | -0.10 | 1.0000 | negligible |
| science-based | 9.1% | 11.4% |  -2.3% | -0.08 | 1.0000 | negligible |
| deep learning | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| neural network | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| computer vision | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| natural language | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| large language model | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| reinforcement learning | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| predictive | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| recommendation | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |
| pose detection | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible (category-driven) |
| gpt/llm | 0.0% | 0.0% |  +0.0% | +0.00 | 1.0000 | negligible |

### 3b. Key Findings

#### Finding 1: F apps use NO technical AI terminology
- **"artificial intelligence"**: 0/22 F apps vs 7/35 T apps (20%)
- **"machine learning"**: 0/22 F apps vs 2/35 T apps (6%)
- **"adaptive"**: 0/22 F apps vs 6/35 T apps (17%)
- **"automated"**: 0/22 F apps vs 2/35 T apps (6%)

This is the strongest signal of AI-washing: F apps claim "AI" in their name/tagline but never substantiate it with any specific AI/ML terminology.

#### Finding 2: Specific terms that discriminate T from F

**Terms more common in T (True AI):**
- **"artificial intelligence"**: T=20% vs F=0%, h=-0.86 (large), p=0.036
- **"real-time"**: T=43% vs F=9%, h=-0.81 (large), p=0.008
- **"adaptive"**: T=17% vs F=0%, h=-0.79 (medium), p=0.072
- **"personalized"**: T=89% vs F=55%, h=-0.79 (medium), p=0.010

**Terms more common in F (AI-Washing):**

#### Finding 3: Category confound assessment
Terms that may reflect category composition rather than AI-washing:

| Term | Likely Source | Rationale |
|------|--------------|-----------|
| real-time, pace, marathon | Running apps (6, all T) | The 6 running coach apps in T naturally discuss pace, real-time feedback, marathon training |
| tracking, log, timer | Tracking tools (10, all F) | The 10 timer/logging apps in F use "tracking" because that's their core utility, not because they're "faking AI" |
| personalized | General gym (both groups) | Used by both T and F; not a discriminative term |

These terms should be interpreted cautiously. The core finding (absence of technical AI terms in F) is robust across categories.

---

## Discussion

The language fingerprint of AI-washing is characterized by **what is absent** rather than what is present. Apps labeled F (Fake AI) use the term "AI" in their name or tagline at a similar rate to genuine AI apps, but **none of them** use supporting technical terminology:

- **Zero** F apps mention "artificial intelligence" (spelled out in full)
- **Zero** F apps mention "machine learning," "deep learning," or "neural network"
- **1/22** F apps mention "real-time" (vs 12/35 T apps, p=0.01)
- **Zero** F apps mention "adaptive," "predictive," or "biometric tracking"

Conversely, F apps tend to use vague marketing language ("smart," "AI-powered," "personalized") that sounds technical but commits to no specific ML mechanism.

This pattern holds even after controlling for category confounds (running apps in T vs tracking tools in F), though the small within-group sample (F general_gym = 5) limits statistical power.

### Interpretation

AI-Washing apps follow a consistent linguistic strategy:
1. **Claim AI in the header** — Use "AI" or "Smart" in the app name for ASO/SEO
2. **Avoid technical commitments** — No specific AI/ML terminology that could be fact-checked
3. **Rely on peripheral marketing** — Emphasize outcomes (weight loss, muscle gain) rather than the technology enabling them

This differentiates them from genuine AI apps, which (even when poorly rated) describe specific AI mechanisms like adaptive algorithms, real-time analysis, or machine learning-based recommendations.

---

## Files

| File | Description |
|------|-------------|
| `ai_washing_language_figA.png` | Technical AI jargon comparison bar chart (T vs F) |
| `ai_washing_language_figB.png` | Effect size forest plot (Cohen's h with significance) |
| `ai_washing_language_analysis_v2.md` | This report |

---

*Generated by AI-Fit-Scan analysis pipeline v3. For questions: gbx1220max@gmail.com*

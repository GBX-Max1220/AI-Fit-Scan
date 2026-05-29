---
license: cc-by-4.0
task_categories:
  - text-classification
  - tabular-classification
language:
  - en
tags:
  - ai-fitness
  - ai-washing
  - mobile-apps
  - google-play
  - health-fitness
  - annotation
size_categories:
  - n<1k
---

# AI-Fit-Scan: A Labeled Dataset of AI-Powered Fitness Applications on Google Play

## Dataset Description

- **Homepage:** https://huggingface.co/datasets/MaxGuo/ai-fit-scan
- **Repository:** https://github.com/GBX-Max1220/AI-Fit-Scan
- **Paper:** *Coming soon*
- **Point of Contact:** Max Guo (gbx1220max@gmail.com)

### Summary

AI-Fit-Scan is a manually annotated dataset of **161 fitness-related mobile applications** scraped from Google Play, with **61 apps claiming AI capabilities** receiving fine-grained human annotation for AI authenticity. We introduce a three-tier labeling framework (True AI / Quasi-AI / Fake AI) and achieve **inter-annotator agreement of κ = 0.856 (Cohen's Kappa)**, indicating almost perfect agreement.

**Key finding:** Among 61 apps claiming to be "AI-powered fitness" applications, **36.1% (22) are fake AI** — their core functionality does not use AI/ML, and **only 57.4% (35) are confirmed true AI**.

### Why This Dataset Matters

1. **AI-Washing quantification:** First systematic measurement of AI claim inflation in the fitness app market
2. **Reusable annotation framework:** Three-tier (T/Q/F) labeling methodology with proven reliability
3. **AI function taxonomy:** Six-category classification of AI capabilities in fitness apps
4. **Academic utility:** Benchmark for studying AI claim verification, consumer deception, and health app regulation

---

## Dataset Structure

### Files

| File | Description | Rows |
|------|-------------|------|
| `ai_fit_scan_full.csv` | Complete 161-app dataset with L1 metadata labels | 161 |
| `ai_fit_scan_annotated.csv` | 61 AI-claiming apps with L2 human annotation | 61 |

### Columns

**Both files:**
- `name` — App name
- `category` — Google Play category
- `installs` — Download count
- `rating` — User rating (0-5)
- `reviews` — Number of user reviews
- `description` — App description from Google Play
- `l1_label` — L1 automatic classification (AI_FITNESS / FITNESS_NO_AI / EXCLUDE_GENERIC / AI_NOT_FITNESS / UNRELATED)

**Annotated file only:**
- `l2_label` — L2 human annotation (T / Q / F)
- `ai_function` — AI capability category (see taxonomy below)
- `annotator_a` — Annotator A label
- `annotator_b` — Annotator B label
- `l2_evidence` — Evidence for L2 label (source URL or rationale)

### Label Definitions

#### L1: Metadata-Based Classification

| Label | Definition | Count |
|-------|-----------|-------|
| AI_FITNESS | Health/Fitness category + AI keywords in description | 61 |
| FITNESS_NO_AI | Health/Fitness category, no AI keywords | 50 |
| EXCLUDE_GENERIC | Generic AI tools (chatbots, translators, etc.) | 25 |
| AI_NOT_FITNESS | AI keywords but non-fitness category | 13 |
| UNRELATED | Neither fitness nor AI | 12 |

#### L2: Human Annotation of AI Authenticity

| Label | Definition | Count | % |
|-------|-----------|-------|---|
| **T (True AI)** | AI/ML is the core driver of the app's primary functionality. Removing AI would fundamentally change the product. | 35 | 57.4% |
| **Q (Quasi-AI)** | App claims AI but evidence is inconclusive. May use rule-based algorithms or simple heuristics marketed as AI. | 4 | 6.6% |
| **F (Fake AI)** | AI label is marketing only. Core functionality works without AI, or "AI" refers to basic automation/features unrelated to ML. | 22 | 36.1% |

**Annotation criteria:**
- **T:** Description explicitly describes AI/ML-driven personalization, adaptive planning, computer vision, or LLM-based coaching as a core feature
- **Q:** App name or description claims "AI" but lacks specific mechanism description; could be rule-based
- **F:** "Smart"/"AI" used as marketing buzzword; core is workout logging, pre-set routines, or timer functionality; AI only used for peripheral features (e.g., food photo recognition in a calorie counter)

### AI Function Taxonomy

| Category | Description | Count |
|----------|-------------|-------|
| Plan Generation | AI generates/adapts personalized workout plans based on user data | 28 |
| LLM Chat Coach | LLM-powered conversational coaching interface | 8 |
| Nutrition AI | AI-driven diet and nutrition recommendations | 6 |
| Pose/Motion Detection | Computer vision for real-time form checking and rep counting | 5 |
| Wearable Integration | AI adjusts plans based on biometric data (HRV, sleep, etc.) | 4 |
| Voice Coach | Real-time AI voice guidance during workouts | 3 |

---

## Data Collection

### Methodology

1. **Search strategy:** 10 keyword queries on Google Play (`"AI fitness"`, `"AI workout"`, `"AI personal trainer"`, `"AI gym"`, `"AI exercise"`, `"AI coach"`, `"AI training plan"`, `"AI running"`, `"smart fitness"`, `"AI health"`)
2. **Deduplication:** Removed duplicate entries by package name
3. **Collection date:** May 2026
4. **Data extracted:** App name, category, installs, rating, reviews, description, developer, price, version history

### Annotation Process

1. **L1 automatic classification:** Keyword-based filtering using AI-related terms and category matching
2. **L2 human annotation:** Two annotators independently labeled all 61 AI_FITNESS apps
   - Annotator A: AI research assistant (automated labeling + L2 web search verification)
   - Annotator B: Domain expert (CSCS-certified, HCI researcher)
3. **L2 evidence collection:** For ambiguous cases (initially labeled Q), web search was conducted to verify AI technology claims against official websites, technical documentation, and third-party reviews
4. **Disagreement resolution:** 5 disagreements (all Q vs T/F) resolved by adopting Annotator B's judgment after reviewing app screenshots and user reviews

### Inter-Annotator Agreement

| Metric | Value |
|--------|-------|
| Raw agreement | 56/61 = 91.8% |
| Cohen's Kappa | **0.856** |
| Interpretation | Almost Perfect (Landis & Koch, 1977) |

---

## Key Findings

### 1. AI-Washing is Rampant
36.1% of apps claiming "AI" in the fitness space have no meaningful AI in their core functionality. The most common fake-AI patterns:
- "Smart" = basic algorithm/timer (e.g., Smart Workout Counter = interval timer)
- "AI-Powered" added to existing traditional apps (e.g., MyFitnessPal adding AI food recognition)
- "Personal Trainer" marketing without any ML (e.g., Home Workout - No Equipment)

### 2. Fake AI Apps Have Higher Ratings
| Group | Mean Rating | Median Rating |
|-------|-------------|---------------|
| True AI (T) | 4.28 | 4.50 |
| Fake AI (F) | 4.41 | 4.54 |

**High-rating apps sell outcome promises; low-rating apps sell AI technology.**

### 3. Long Tail of Irrelevance
- 28 of 61 "AI fitness" apps have <50K downloads
- 15 apps have zero or missing ratings — likely inactive or abandoned
- The market has not consolidated; room for genuine AI entrants

### 4. AI Function Distribution
Plan generation dominates (28/35 true-AI apps), while pose detection (5) and voice coaching (3) remain underdeveloped despite high user value potential.

---

## Limitations

1. **Single platform:** Google Play only; Apple App Store data not included
2. **Single time point:** Data collected May 2026; AI claims may change with updates
3. **Description-dependent:** L2 annotation relies primarily on app descriptions and public documentation, not source code inspection
4. **Binary framing:** The T/Q/F taxonomy simplifies a spectrum; some "Quasi-AI" apps may use sophisticated rule engines that approach ML-level personalization
5. **Keyword bias:** Search results favor apps with "AI" in their name/description; genuine AI apps using different marketing language may be missed

---

## Ethical Considerations

- App names and descriptions are publicly available Google Play metadata
- Developer contact and revenue data are **not** included
- We do not accuse any app of fraud; "Fake AI" refers to absence of ML in core functionality, not malicious intent
- Apps labeled F may use AI in peripheral features (e.g., food recognition, recommendation systems)

---

## Citation

```bibtex
@dataset{guo2026aifitscan,
  title={AI-Fit-Scan: A Labeled Dataset of AI-Powered Fitness Applications on Google Play},
  author={Guo, Baixin (Max) and MaxCoze},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/MaxGuo/ai-fit-scan}
}
```

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)

## Acknowledgments

- Cal Dietz Triphasic Training system — conceptual influence on training periodization analysis
- Google Play Store — public data source

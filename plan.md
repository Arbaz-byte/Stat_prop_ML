# Complete June 2026 Plan — Month 2 of 12

**Theme:** From mathematical foundations to the learning algorithms that power all of AI.  
**Core Rule:** For every lecture watched, immediately open a notebook and implement the concept. Push daily.  
**Hardware:** Avita Pura (Core i3, 4GB) — small matrices, small datasets, plenty of CPU power for what we need.

## The Big Picture for June

You will run **two parallel threads** all month:

| Thread Focus                  | Source                          | Purpose                          |
|-------------------------------|---------------------------------|----------------------------------|
| **Thread A (Mornings)**      | Calculus → Gradient Descent → Autodiff | Original Month 2 plan + CS229 lectures |
| **Thread B (Afternoons/Evenings)** | Probability & Statistics for ML | Decode AiML playlist (curated) + CS229 |

- Every **Saturday** is project day (mini or major).  
- Every **Sunday** is consolidation, LinkedIn post, or rest.

## CS229 Integration Strategy

We will extract **Lectures 1–5** only (aligned with June topics):

- Lecture 1: Intro, supervised learning, linear regression
- Lecture 2: Locally weighted regression, probabilistic interpretation, logistic regression
- Lecture 3: Perceptron, exponential family, GLMs
- Lecture 4: Generative learning algorithms (GDA, Naive Bayes)
- Lecture 5: Naive Bayes, Laplace smoothing, event models

**Order:** Watch CS229 segment → Code corresponding math from Decode AiML → Mini project.

---

## Week-by-Week Breakdown

### Week 1 (June 1–7) — Calculus & Gradient Descent + Probability Intro

| Day | Thread A: Calculus (Library, 12–4pm) | Thread B: Probability (Evenings, 9–10pm) | CS229 Push |
|-----|---------------------------------------|-------------------------------------------|------------|
| Mon | Derive MSE gradient on paper. NumPy gradient function. | Playlist intro + 10.1–10.2 | Mental map |
| Tue | Full-batch GD for linear regression + loss curve animation | 10.3–10.4 (counting, permutations) | Lecture 1 (first 40 min) |
| Wed | Experiment with learning rates (0.001, 0.01, 0.1, 1) | 10.5–10.6 (permutation problems) | — |
| Thu | Stochastic/mini-batch GD | 10.7 (Intro to probability) | — |
| Fri | Buffer / catch-up. Add momentum (conceptual) | — | Lecture 2 |
| Sat | **Mini Project 1**: Linear regression (GD + normal equation) on real/synthetic data | — | `project1_linear_regression.ipynb` |
| Sun | Rest + LinkedIn post about gradient descent | — | Post |

### Week 2 (June 8–14) — Autodiff & Micrograd + Probability Rules

| Day | Thread A: Autodiff | Thread B: Probability | CS229 Push |
|-----|--------------------|-----------------------|------------|
| Mon | Karpathy micrograd video → implement `Value` class | 10.8–10.9 (events, axioms, independence) | `09_micrograd.ipynb` |
| Tue | Complete `Value`: add, mul, backward() | 10.10–10.11 (conditional probability) | Lecture 3 (first half) |
| Wed | Tiny 2-layer net on XOR | 10.12–10.14 (total prob, Bayes) + code medical test example | `10_bayes_medical.py` |
| Thu | Compare micrograd vs PyTorch autograd on XOR | 10.15 (Naive Bayes) | Lecture 4 |
| Fri | Buffer / polish | 10.16–10.17 (population & sample) | — |
| Sat | **Mini Project 2**: Naive Bayes spam classifier from scratch | — | `project2_naive_bayes.ipynb` |
| Sun | Rest + LinkedIn post (neural net from scratch) | — | Post |

### Week 3 (June 15–21) — Probability Deep Dive (Descriptive Stats, Random Variables)

| Day | Thread A: Calculus Maintenance | Thread B: Probability (Primary) | CS229 Push |
|-----|--------------------------------|---------------------------------|------------|
| Mon | Add L2 regularization to GD notebook | 10.18–10.19 (mean, median, mode) | Lecture 5 |
| Tue | (Light) Read loss functions (MSE, MAE, cross-entropy) | 10.20–10.21 (variance, std, Bessel) | — |
| Wed | — | 10.22–10.24 (IQR, percentiles, box plots) | `12_iqr_boxplot.ipynb` |
| Thu | — | 10.25–10.28 (Random variables, expectation, PMF/CDF) | `13_random_variables.ipynb` |
| Fri | — | 10.29–10.31 (Continuous PDF/CDF) | — |
| Sat | **Mini Project 3**: Central Limit Theorem simulation | — | `project3_clt_simulation.ipynb` |
| Sun | Rest + LinkedIn post (dice → CLT) | — | Post |

### Week 4 (June 22–30) — Distributions, MLE, and Inferential Statistics

| Day | Thread A: Bridge to Month 3 | Thread B: Probability & Stats | CS229 Push |
|-----|-----------------------------|-------------------------------|------------|
| Mon | PyTorch tutorial (tensors) sandbox | 10.32–10.35 (Common distributions) | `14_distributions.ipynb` |
| Tue | — | 10.36–10.38 (Exponential, Normal, Z-score) | — |
| Wed | — | 10.39–10.42 (T-dist, Chi-square, correlation) | `15_correlation_moments.ipynb` |
| Thu | — | 10.43–10.46 (Sampling, LLN, CLT, MLE) | `16_mle.ipynb` |
| Fri | — | 10.47–10.54 (Confidence intervals, hypothesis tests) | `17_hypothesis_tests.ipynb` |
| Sat | **Major Project (Capstone)**: Full ML pipeline (EDA + model + eval) on Iris/Titanic/etc. | — | `project4_full_pipeline.ipynb` + blog |
| Sun | Month 2 Closure: Review notebooks, update README, detailed LinkedIn article | — | Article + push |

---

## Extra Mini & Major Projects for June

| Project Type | When | Description |
|--------------|------|-----------|
| Linear regression (GD + normal eq.) | Week 1 Sat | Compare on housing dataset |
| Naive Bayes spam classifier | Week 2 Sat | From scratch, text data |
| Central Limit Theorem simulation | Week 3 Sat | Animated histograms |
| Full ML pipeline | Week 4 Sat | **Major** — EDA + model + report (portfolio piece) |
| (Optional) MLE vs MAP | Week 4 | Prior effect on estimation |

## CS229 Integration Summary

| Lecture | Watched During | Topic Links |
|---------|----------------|-------------|
| 1 | Week 1 Tue | Linear regression |
| 2 | Week 1 Fri | Probabilistic interpretation |
| 3 (first half) | Week 2 Tue | Logistic, exponential family |
| 4 | Week 2 Thu | Generative models, Naive Bayes |
| 5 | Week 3 Mon | Laplace smoothing |

---

## What You Will NOT Do in June

- Finish all 58 probability lectures (curated only)
- Start GenAI projects (math first)
- Deep Pandas (basic CSV only)
- Deploy APIs (August)
- Compare yourself to “no-math GenAI” people

**By July 1 you will have:**
- Working GD + autodiff engine
- Deep coded understanding of probability & stats
- 4 mini + 1 major project
- Strong GitHub + LinkedIn presence
- First 5 CS229 lectures completed



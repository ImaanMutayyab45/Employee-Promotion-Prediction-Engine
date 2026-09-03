# Employee Promotion Prediction Engine

### The Problem
Promotion decisions are often inconsistent -influenced by manager bias, visibility, or who asks, rather than purely by performance and readiness. Employees who are actually promotion-ready sometimes get overlooked, while HR lacks a systematic, data-backed way to flag who's ready before the annual review cycle.

### The Goal
Build a classification model that predicts whether an employee will be promoted, based on their performance ratings, training scores, tenure, and past achievements - using real historical promotion outcomes as ground truth (not a guessed or circular label, unlike the earlier department-fit idea).

### Why This Is Helpful
- Gives HR an objective, data-driven shortlist of promotion-ready employees ahead of review cycles, reducing reliance on manager visibility/bias alone.
- Surfaces employees who are strong performers but might be getting passed over.
- The "readiness score" framing is directly actionable - unlike a vague fit score, this ties to a real, scheduled business decision (the promotion cycle).

### How Is This Better Than Just Asking GPT?
- GPT can describe general promotion criteria, but can't learn what actually predicted promotion in this specific company's real historical data - that pattern only comes from training on real outcomes.
- You get calibrated, per-employee probabilities - a ranked list HR can act on directly, not a general opinion.
- Feature importance shows *why* someone is flagged as promotion-ready (e.g., training score + KPI met %), which is auditable and defensible - important since promotion decisions carry fairness/compliance weight.

### Which Sector
HR / people analytics - applicable across any large organization with structured performance review data.

### Dataset
**HR Analytics: Employee Promotion** - [Kaggle link](https://www.kaggle.com/datasets/arashnic/hr-ana)

~54,808 employees, with real fields including: department, education, previous year rating, KPIs met (>80%), awards won, average training score, length of service, and the actual target column `is_promoted`.

### Pipeline

1. **EDA:** promotion rate overall and by department, class imbalance check (promotions are usually a small minority - important to flag early), correlation between training score/KPI/awards and promotion, distribution of tenure among promoted vs. non-promoted.
2. **Feature Engineering:** KPI-met flag, awards-won flag, training score bucket, tenure-to-promotion-cycle ratio, previous rating trend (if multiple years available), education level encoding.
3. **Modeling:** Logistic Regression baseline → Random Forest main model (handle class imbalance via class weights, since promoted employees are a small minority - this is the most important technical decision in this project).
4. **Evaluation:** precision/recall/F1 (recall and precision both matter here - recall so you don't miss ready employees, precision so you don't flood HR with false positives), confusion matrix, ROC-AUC, feature importance.
5. **Dashboard:** employee lookup showing promotion-readiness probability + top contributing factors; department-wide ranked list of promotion-ready candidates; a simple "what would improve this employee's odds" indicator (e.g., "training score is the biggest gap").

# Employee Promotion Prediction Engine
A classification project that predicts whether an employee will be promoted, using real historical HR data: performance ratings, training scores, tenure, and past achievements - as ground truth. Built for HR/people analytics teams who need an objective, auditable shortlist of promotion-ready employees ahead of review cycles.

### Problem Statement
Promotion decisions inside most organizations are inconsistent - shaped by manager bias, visibility, or who advocates loudest for themselves, rather than purely by performance and readiness. This project builds a model that learns what actually predicted promotion in real historical outcomes, giving HR a data-backed, defensible signal instead of a guess.

### Dataset
**Source:** [HR Analytics: Employee Promotion (Kaggle)](https://www.kaggle.com/datasets/arashnic/hr-ana)
| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 54,808 | 13 | Includes target column `is_promoted` |
| `test.csv` | 23,490 | 12 | Same schema minus `is_promoted`; used for held-out predictions |

**Columns:** `employee_id`, `department`, `region`, `education`, `gender`, `recruitment_channel`, `no_of_trainings`, `age`, `previous_year_rating`, `length_of_service`, `awards_won?`, `avg_training_score`, `is_promoted` (train only)

**Known data quirks:**
- `education` has 2,409 missing values (no clear structural pattern — mode imputation is reasonable).
- `previous_year_rating` has 4,124 missing values, all corresponding to employees with `length_of_service == 1` (i.e., new hires who haven't had a review cycle yet — impute with a sentinel value like `0`, not the median).
- This version of the dataset does **not** include a `KPIs_met >80%` column, unlike some public mirrors.
- The target class is heavily imbalanced: only **8.5%** of employees in `train.csv` were promoted. Any model must account for this (e.g. `class_weight='balanced'`), or accuracy alone will be a misleading metric.

### Project Pipeline

1. **EDA:** promotion rate overall and by department, class imbalance check, correlation between training score/KPI/awards and promotion, tenure distribution split by promotion outcome.
2. **Feature Engineering:** KPI-met flag (pending data availability), awards-won flag, training score buckets, tenure-to-promotion-cycle ratio, previous rating trend (if multi-year data available), education level encoding.
3. **Modeling:** Logistic Regression baseline → Random Forest main model, with class weighting to handle the imbalance.
4. **Evaluation:**  precision, recall, F1, confusion matrix, ROC-AUC, feature importance.
5. **Dashboard:** employee lookup with promotion-readiness probability and top contributing factors, department-wide ranked candidate list, and a "biggest gap" indicator per employee.

### 📂 Repository Structure

```text
.
├── train.csv                                     # Raw training data with is_promoted labels
├── test.csv                                      # Raw held-out data for predictions
├── cleaned_promotion_data.csv                    # Output of Section 2: cleaned, deduplicated data
├── preprocessed_promotion_data.csv               # Output of Section 4: feature-engineered data
├── Employee-Promotion-Prediction-Engine-FINAL.ipynb  # Final notebook: EDA, feature engineering, modeling, dashboard code
├── app.py                                        # Streamlit dashboard app
├── model.pkl                                     # Trained Random Forest model (stored via Git LFS)
├── model_columns.pkl                             # Column order expected by the model (stored via Git LFS)
├── requirements.txt                              # Python dependencies for running app.py / Streamlit Cloud
├── confusion_matrix_baseline.png                 # Logistic Regression confusion matrix
├── confusion_matrix_rf.png                       # Random Forest confusion matrix
├── feature_importance_baseline.png               # Logistic Regression top feature coefficients
├── feature_importance_rf.png                     # Random Forest top feature importances
├── roc_curve_baseline.png                        # Logistic Regression ROC curve
├── roc_curve_rf.png                              # Random Forest ROC curve
├── Capstone-Project-Outline.md                   # Full project brief
├── .gitattributes                                # Marks model.pkl / model_columns.pkl for Git LFS
└── README.md                                     # This file
```
### Team Roles
| Member | Responsibility |
|---|---|
| Imaan Mutayyab | Data acquisition, cleaning, and exploratory data analysis |
| Nimra Farooq | Feature engineering | Baseline Modeling
| Sofia Rashid Abdul Rashid| Modeling (Logistic Regression + Random Forest, class imbalance handling) |
| Nimra Farooq & Imaan Mutayyab | Dashboard / employee lookup tool |

### Why This Approach (vs. Just Asking an LLM)
A general-purpose language model can describe typical promotion criteria, but it can't learn what actually predicted promotion in this company's specific historical data. This project instead produces calibrated, per-employee probabilities and feature-importance explanations that are auditable and defensible - important given the fairness and compliance weight that promotion decisions carry.

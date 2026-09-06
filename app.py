import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest and the exact feature-column order it was trained on (from Section 5)
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Load the SAME feature-engineered dataset the model was trained on (not the raw cleaned data —
# see the bug-fix note above) so the app's inputs match what the model actually learned from.
df = pd.read_csv('preprocessed_promotion_data.csv')


def prepare_input(raw_df):
    """Encode a slice of the feature-engineered dataframe the same way Section 5 did,
    then align its columns to exactly what the model expects (fills any missing dummy
    columns with 0, e.g. a category that doesn't appear in a single selected row)."""
    encoded = pd.get_dummies(
        raw_df, columns=raw_df.select_dtypes(include=['object', 'category']).columns.tolist(),
        drop_first=True
    )
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    return encoded


st.title("Employee Promotion Predictor")

tab1, tab2, tab3 = st.tabs(["Employee Lookup", "Department Rankings", "What-If Analysis"])

with tab1:
    emp_id = st.selectbox("Select Employee ID", df['employee_id'].unique())
    emp_row = df[df['employee_id'] == emp_id].drop(columns=['is_promoted', 'employee_id'], errors='ignore')
    X_input = prepare_input(emp_row)
    proba = model.predict_proba(X_input)[0][1]
    st.metric("Promotion Probability", f"{proba:.1%}")
    st.dataframe(emp_row)

with tab2:
    depts = df['department'].unique()
    dept_scores = {}
    for dept in depts:
        subset = df[df['department'] == dept].drop(columns=['is_promoted', 'employee_id'], errors='ignore')
        X_dept = prepare_input(subset)
        dept_scores[dept] = model.predict_proba(X_dept)[:, 1].mean()
    scores_series = pd.Series(dept_scores).sort_values(ascending=False)
    st.bar_chart(scores_series)

with tab3:
    st.write("Adjust values to see how promotion probability changes")

    whatif_emp_id = st.selectbox("Select Employee ID for simulation", df['employee_id'].unique(), key="whatif_emp")
    sample = df[df['employee_id'] == whatif_emp_id].drop(columns=['is_promoted', 'employee_id'], errors='ignore').copy()

    # Show the rest of this employee's profile, so it's clear the probability below
    # reflects "this specific employee, but with a different training score" — not a generic average.
    st.caption("This employee's other traits (only Training Score is changed below):")
    display_cols = [c for c in ['department', 'previous_year_rating', 'awards_won_flag',
                                 'length_of_service', 'age', 'education_encoded'] if c in sample.columns]
    st.dataframe(sample[display_cols], hide_index=True)

    default_score = int(sample['avg_training_score'].iloc[0]) if 'avg_training_score' in sample.columns else 60
    training_score = st.slider("Training Score", 0, 100, default_score)
    if 'avg_training_score' in sample.columns:
        sample['avg_training_score'] = training_score

    if 'training_score_bucket' in sample.columns:
        sample['training_score_bucket'] = pd.cut(
            [training_score], bins=[-1, 40, 60, 80, 100],
            labels=["low", "medium", "high", "very_high"]
        )[0]
    X_sample = prepare_input(sample)
    whatif_proba = model.predict_proba(X_sample)[0][1]
    st.metric("What-If Promotion Probability", f"{whatif_proba:.1%}")
    st.caption(
        "This is only the effect of training score, holding this employee's other traits "
        "(rating, awards, tenure, etc. shown above) fixed — a different employee would give "
        "a different starting point."
    )

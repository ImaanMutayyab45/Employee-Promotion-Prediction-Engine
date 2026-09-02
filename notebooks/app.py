import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')
df = pd.read_csv('cleaned_promotion_data.csv')  # match your actual raw filename

def prepare_input(raw_df):
    encoded = pd.get_dummies(raw_df, drop_first=True)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    return encoded

st.title("Employee Promotion Predictor")

tab1, tab2, tab3 = st.tabs(["Employee Lookup", "Department Rankings", "What-If Analysis"])

with tab1:
    emp_id = st.selectbox("Select Employee ID", df['employee_id'].unique())
    emp_row = df[df['employee_id'] == emp_id].drop(columns=['is_promoted'], errors='ignore')
    X_input = prepare_input(emp_row)
    proba = model.predict_proba(X_input)[0][1]
    st.metric("Promotion Probability", f"{proba:.1%}")
    st.dataframe(emp_row)

with tab2:
    depts = df['department'].unique()
    dept_scores = {}
    for dept in depts:
        subset = df[df['department'] == dept].drop(columns=['is_promoted'], errors='ignore')
        X_dept = prepare_input(subset)
        dept_scores[dept] = model.predict_proba(X_dept)[:, 1].mean()
    scores_series = pd.Series(dept_scores).sort_values(ascending=False)
    st.bar_chart(scores_series)

with tab3:
    st.write("Adjust values to see how promotion probability changes")
    training_score = st.slider("Training Score", 0, 100, 60)
    sample = df.drop(columns=['is_promoted'], errors='ignore').iloc[[0]].copy()
    if 'avg_training_score' in sample.columns:
        sample['avg_training_score'] = training_score
    X_sample = prepare_input(sample)
    whatif_proba = model.predict_proba(X_sample)[0][1]
    st.metric("What-If Promotion Probability", f"{whatif_proba:.1%}")

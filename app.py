import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Wide layout so the dashboard fills the screen instead of a narrow centered column
st.set_page_config(page_title="Employee Promotion Predictor", layout="wide", page_icon="🏆")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; padding: 8px 4px; }
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 14px 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# Load the trained Random Forest and the exact feature-column order it was trained on (from Section 5)
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Load the SAME feature-engineered dataset the model was trained on (not the raw cleaned data),
# so the app's inputs match what the model actually learned from.
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


@st.cache_data
def score_all_employees(_df):
    """Runs every employee through the model once, so every tab can reuse the same scores
    instead of recomputing predictions repeatedly. Cached so it only runs once per session."""
    X_all = prepare_input(_df.drop(columns=['is_promoted', 'employee_id'], errors='ignore'))
    scored = _df.copy()
    scored['promotion_probability'] = model.predict_proba(X_all)[:, 1]
    return scored


df_scored = score_all_employees(df)

st.title("🏆 Employee Promotion Predictor")
st.caption("Data-driven promotion-readiness scoring, built on real historical HR outcomes.")

tab_overview, tab_lookup, tab_dept, tab_whatif = st.tabs(
    ["📊 Overview", "🔍 Employee Lookup", "🏢 Department Rankings", "🎯 What-If Analysis"]
)

# ======================================================================
# TAB: OVERVIEW — the new landing-page summary
# ======================================================================
with tab_overview:
    st.subheader("Workforce Summary")

    col_t1, col_t2 = st.columns(2)
    ready_threshold = col_t1.slider("Promotion-ready threshold (%)", 10, 90, 50, step=5) / 100
    effort_threshold = col_t2.slider("\"Needs significant improvement\" threshold (%)", 5, 50, 20, step=5) / 100

    total_employees = len(df_scored)
    ready_count = (df_scored['promotion_probability'] >= ready_threshold).sum()
    effort_pct = (df_scored['promotion_probability'] < effort_threshold).mean() * 100
    avg_prob = df_scored['promotion_probability'].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{total_employees:,}")
    c2.metric("Promotion-Ready", f"{ready_count:,}", f"{ready_count/total_employees:.1%} of workforce")
    c3.metric("Need Significant Improvement", f"{effort_pct:.1f}%")
    c4.metric("Avg. Promotion Probability", f"{avg_prob:.1f}%")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🌟 Top Performers")
        top_n = st.slider("How many top employees to show", min_value=3, max_value=20, value=5)
        top_employees = df_scored.sort_values('promotion_probability', ascending=False).head(top_n)
        show_cols = [c for c in ['employee_id', 'department', 'avg_training_score',
                                  'previous_year_rating', 'promotion_probability'] if c in top_employees.columns]
        top_display = top_employees[show_cols].copy()
        top_display['promotion_probability'] = (top_display['promotion_probability'] * 100).round(1).astype(str) + '%'
        st.dataframe(top_display, hide_index=True, use_container_width=True)

    with col_right:
        st.markdown("#### 📈 Promotion Probability Distribution")
        fig_dist = px.histogram(
            df_scored, x='promotion_probability', nbins=30,
            labels={'promotion_probability': 'Promotion Probability'},
            color_discrete_sequence=['#FF4B4B']
        )
        fig_dist.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.markdown("#### 🏢 Average Promotion Probability by Department")
    dept_avg = (df_scored.groupby('department')['promotion_probability'].mean().sort_values(ascending=False) * 100)
    fig_dept = px.bar(
        dept_avg, orientation='h',
        labels={'value': 'Avg. Promotion Probability (%)', 'department': ''},
        color=dept_avg.values, color_continuous_scale='Blues'
    )
    fig_dept.update_layout(height=380, coloraxis_showscale=False, showlegend=False)
    st.plotly_chart(fig_dept, use_container_width=True)

# ======================================================================
# TAB: EMPLOYEE LOOKUP — your original dropdown, kept as its own tab
# ======================================================================
with tab_lookup:
    st.subheader("🔍 Individual Employee Lookup")

    emp_id = st.selectbox("Select Employee ID", df_scored['employee_id'].unique())
    emp_row = df_scored[df_scored['employee_id'] == emp_id]
    proba = emp_row['promotion_probability'].values[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Promotion Probability", f"{proba:.1%}")
        if proba >= 0.5:
            st.success("Promotion-ready")
        elif proba >= 0.2:
            st.warning("Developing — needs improvement")
        else:
            st.error("Needs significant improvement")

    with col2:
        display_cols = [c for c in ['department', 'region', 'education', 'previous_year_rating',
                                     'avg_training_score', 'awards_won_flag', 'length_of_service', 'age']
                         if c in emp_row.columns]
        st.dataframe(emp_row[display_cols], hide_index=True, use_container_width=True)

# ======================================================================
# TAB: DEPARTMENT RANKINGS
# ======================================================================
with tab_dept:
    st.subheader("🏢 Department Rankings")

    dept_scores = df_scored.groupby('department')['promotion_probability'].mean().sort_values(ascending=False) * 100
    fig_dept2 = px.bar(
        dept_scores, orientation='h',
        labels={'value': 'Avg. Promotion Probability (%)', 'department': ''},
        color=dept_scores.values, color_continuous_scale='Blues'
    )
    fig_dept2.update_layout(height=420, coloraxis_showscale=False, showlegend=False)
    st.plotly_chart(fig_dept2, use_container_width=True)

    st.markdown("#### Department Breakdown")
    dept_table = df_scored.groupby('department').agg(
        employees=('employee_id', 'count'),
        avg_promotion_probability=('promotion_probability', 'mean'),
        avg_training_score=('avg_training_score', 'mean')
    ).sort_values('avg_promotion_probability', ascending=False)
    dept_table['avg_promotion_probability'] = (dept_table['avg_promotion_probability'] * 100).round(1).astype(str) + '%'
    dept_table['avg_training_score'] = dept_table['avg_training_score'].round(1)
    st.dataframe(dept_table, use_container_width=True)

# ======================================================================
# TAB: WHAT-IF ANALYSIS
# ======================================================================
with tab_whatif:
    st.subheader("🎯 What-If Analysis")
    st.write("Adjust an employee's training score to see how their promotion probability changes.")

    whatif_emp_id = st.selectbox("Select Employee ID for simulation", df['employee_id'].unique(), key="whatif_emp")
    sample = df[df['employee_id'] == whatif_emp_id].drop(columns=['is_promoted', 'employee_id'], errors='ignore').copy()

    st.caption("This employee's other traits (only Training Score is changed below):")
    display_cols = [c for c in ['department', 'previous_year_rating', 'awards_won_flag',
                                 'length_of_service', 'age', 'education_encoded'] if c in sample.columns]
    st.dataframe(sample[display_cols], hide_index=True, use_container_width=True)

    default_score = int(sample['avg_training_score'].iloc[0]) if 'avg_training_score' in sample.columns else 60
    training_score = st.slider("Training Score", 0, 100, default_score)
    if 'avg_training_score' in sample.columns:
        sample['avg_training_score'] = training_score
    # The model relies on training_score_bucket (a derived feature from Section 4.1), not just the raw
    # score, so moving the slider must recompute the bucket too, or the probability barely moves.
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
        "(rating, awards, tenure, etc. shown above) fixed."
    )

    st.divider()
    st.markdown("#### 📊 Training Score Overview (All Employees)")

    high_score_pct = (df['avg_training_score'] >= 80).mean() * 100
    avg_score = df['avg_training_score'].mean()

    colA, colB = st.columns(2)
    colA.metric("Employees with High Training Score (80+)", f"{high_score_pct:.1f}%")
    colB.metric("Average Training Score (all employees)", f"{avg_score:.1f}")

    fig_score = px.histogram(
        df, x='avg_training_score', nbins=25,
        labels={'avg_training_score': 'Average Training Score'},
        color_discrete_sequence=['#4B9EFF']
    )
    fig_score.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig_score, use_container_width=True)

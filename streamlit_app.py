import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from cancer_ml_pipeline import CancerPredictionPipeline, NUMERICAL_COLS
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MediCore Cancer Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:wght@600;700&display=swap');

    :root {
        --primary-color: #f87171;
        --secondary-color: #b91c1c;
        --accent-color: #fca5a5;
        --success-color: #34d399;
        --danger-color: #dc2626;
        --card-bg: rgba(31, 29, 29, 0.85);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --text-primary: #f5f5f5;
        --text-secondary: #d4d4d4;
        --border-color: rgba(248, 113, 113, 0.25);
    }

    * { font-family: 'DM Sans', sans-serif; }

    .main, .stApp {
        background: linear-gradient(135deg, #1c1917 0%, #2d2a27 50%, #4b4b4b 100%);
        color: var(--text-primary);
    }

    .main-header {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.15), rgba(185, 28, 28, 0.15));
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0 2rem 0;
        text-align: center;
    }

    .main-title {
        font-family: 'Fraunces', serif;
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f87171, #b91c1c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .main-subtitle { font-size: 1.15rem; color: var(--text-secondary); margin-top: 0.5rem; }

    .modern-card, .metric-card, .prediction-card, .sidebar-metric, .footer {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .prediction-card {
        border: 2px solid var(--primary-color);
        text-align: center;
        padding: 2rem;
    }

    .risk-high {
        border-color: var(--danger-color) !important;
        background: linear-gradient(135deg, var(--glass-bg), rgba(220, 38, 38, 0.12)) !important;
    }

    .prediction-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
    .prediction-probability { font-size: 2.8rem; font-weight: 800; margin: 0.5rem 0; }
    .prediction-description { color: var(--text-secondary); }

    .metric-value { font-size: 2rem; font-weight: 700; color: var(--primary-color); margin: 0; text-align: center; }
    .metric-label { font-size: 0.9rem; color: var(--text-secondary); text-align: center; text-transform: uppercase; letter-spacing: 1px; }

    .success-badge {
        background: var(--success-color);
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    h1, h2, h3, h4 { color: var(--text-primary) !important; }
    h2 { color: var(--primary-color) !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    required = [
        "cancer_models.pkl",
        "label_encoders.pkl",
        "scaler.pkl",
        "feature_names.pkl",
        "model_metrics.pkl",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        return None, missing
    pipeline = CancerPredictionPipeline().load_artifacts()
    return pipeline, []


@st.cache_data
def load_dataset():
    path = Path("cancer_dataset.csv")
    if path.exists():
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    return None


def build_input_row(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values])


def main():
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">MediCore Cancer Analytics</div>
            <div class="main-subtitle">AI-Powered Cancer Risk Prediction with Live Model Inference</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline, missing = load_pipeline()
    dataset = load_dataset()

    if pipeline is None:
        st.error(
            "Trained models not found: "
            + ", ".join(missing)
            + ". Run `python run_complete_pipeline.py` first."
        )
        st.stop()

    model_scores = pipeline.model_scores
    best_model = pipeline.best_model_name
    best_acc = pipeline.best_accuracy
    metrics = pipeline.model_metrics.get(best_model, {})

    with st.sidebar:
        st.markdown("### Navigation Hub")
        st.markdown(
            f"""
            <div class="sidebar-metric">
                <div class="success-badge">Best Model: {best_model}</div>
                <div style="color:#f87171;font-size:1.5rem;font-weight:700;margin-top:1rem;">
                    {best_acc:.1%}
                </div>
                <div style="color:#d4d4d4;font-size:0.9rem;">Live Accuracy Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        model_choice = st.selectbox(
            "Prediction model",
            list(pipeline.models.keys()),
            index=list(pipeline.models.keys()).index(best_model),
        )
        st.markdown(
            """
            <div class="sidebar-metric">
                <div class="success-badge">Models Loaded</div>
                <div style="margin-top:0.75rem;color:#d4d4d4;font-size:0.9rem;">
                    Real RF / XGBoost / LightGBM / Ensembles
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "AI Prediction Engine",
            "Model Performance",
            "Global Health",
            "Data Insights",
            "Feature Analysis",
        ]
    )

    with tab1:
        st.markdown(
            f"""
            <div class="modern-card">
                <h2>Cancer Risk Assessment</h2>
                <p style="color:#d4d4d4;">
                    Predictions come from trained models (not random). Best ensemble accuracy: {best_acc:.1%}.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            with st.form("prediction_form"):
                c1, c2 = st.columns(2)
                with c1:
                    country = st.selectbox(
                        "Country",
                        [
                            "USA", "UK", "Germany", "France", "Canada", "Australia",
                            "Japan", "India", "Brazil", "China", "Italy", "Spain",
                            "Netherlands", "Sweden", "Norway", "Denmark", "Finland",
                        ],
                    )
                    age = st.slider("Age (years)", 18, 100, 55)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    smoking_status = st.selectbox(
                        "Smoking Status", ["Never", "Former", "Current"]
                    )
                    second_hand_smoke = st.selectbox("Second Hand Smoke", ["No", "Yes"])
                    air_pollution = st.slider("Air Pollution Exposure", 0.0, 10.0, 5.0, 0.1)
                    occupation_exposure = st.selectbox(
                        "Occupational Exposure", ["Low", "Medium", "High"]
                    )
                    rural_urban = st.selectbox("Residential Area", ["Rural", "Urban"])
                    socioeconomic_status = st.selectbox(
                        "Socioeconomic Status", ["Low", "Medium", "High"]
                    )
                    healthcare_access = st.selectbox(
                        "Healthcare Access", ["Poor", "Fair", "Good", "Excellent"]
                    )
                    insurance_coverage = st.selectbox(
                        "Insurance Coverage", ["Uninsured", "Partial", "Full"]
                    )
                    screening = st.selectbox("Screening Availability", ["No", "Yes"])

                with c2:
                    stage = st.slider("Stage at Diagnosis", 1, 4, 2)
                    cancer_type = st.selectbox(
                        "Cancer Type Context",
                        [
                            "Lung", "Breast", "Colorectal", "Prostate", "Liver",
                            "Stomach", "Cervical", "Bladder", "Kidney", "Other",
                        ],
                    )
                    mutation_type = st.selectbox(
                        "Mutation Type",
                        [
                            "No Mutation", "EGFR", "KRAS", "ALK", "ROS1", "BRAF",
                            "PIK3CA", "TP53", "BRCA1", "BRCA2",
                        ],
                    )
                    treatment_access = st.selectbox(
                        "Treatment Access", ["Low", "Medium", "High"]
                    )
                    clinical_trial = st.selectbox("Clinical Trial Access", ["No", "Yes"])
                    language_barrier = st.selectbox("Language Barrier", ["No", "Yes"])
                    mortality_risk = st.slider("Mortality Risk Score", 0.0, 1.0, 0.4, 0.01)
                    survival = st.slider("5-Year Survival Probability", 0.0, 1.0, 0.7, 0.01)
                    delay = st.slider("Delay in Diagnosis (days)", 0, 180, 30)
                    family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
                    indoor_smoke = st.selectbox("Indoor Smoke Exposure", ["No", "Yes"])
                    tobacco_marketing = st.selectbox(
                        "Tobacco Marketing Exposure", ["Low", "Medium", "High"]
                    )

                submitted = st.form_submit_button(
                    "Analyze Cancer Risk", use_container_width=True
                )

        with col2:
            st.markdown(
                """
                <div class="modern-card">
                    <h3 style="color:#f87171;">Risk Assessment Results</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if submitted:
                input_row = build_input_row(
                    {
                        "Country": country,
                        "Age": age,
                        "Gender": gender,
                        "Smoking_Status": smoking_status,
                        "Second_Hand_Smoke": second_hand_smoke,
                        "Air_Pollution_Exposure": air_pollution,
                        "Occupation_Exposure": occupation_exposure,
                        "Rural_or_Urban": rural_urban,
                        "Socioeconomic_Status": socioeconomic_status,
                        "Healthcare_Access": healthcare_access,
                        "Insurance_Coverage": insurance_coverage,
                        "Screening_Availability": screening,
                        "Stage_at_Diagnosis": stage,
                        "Cancer_Type": cancer_type,
                        "Mutation_Type": mutation_type,
                        "Treatment_Access": treatment_access,
                        "Clinical_Trial_Access": clinical_trial,
                        "Language_Barrier": language_barrier,
                        "Mortality_Risk": mortality_risk,
                        "5_Year_Survival_Probability": survival,
                        "Delay_in_Diagnosis": delay,
                        "Family_History": family_history,
                        "Indoor_Smoke_Exposure": indoor_smoke,
                        "Tobacco_Marketing_Exposure": tobacco_marketing,
                    }
                )

                prediction, probability = pipeline.predict(
                    input_row, model_name=model_choice
                )
                risk_probability = float(probability[0][1])
                pred_label = int(prediction[0])

                risk_class = "risk-high" if pred_label == 1 else ""
                status_text = "HIGH RISK" if pred_label == 1 else "LOW RISK"
                status_color = "#dc2626" if pred_label == 1 else "#34d399"
                recommendation = (
                    "Immediate medical consultation recommended"
                    if pred_label == 1
                    else "Continue regular health screenings"
                )

                st.markdown(
                    f"""
                    <div class="prediction-card {risk_class}">
                        <div class="prediction-title" style="color:{status_color};">{status_text}</div>
                        <div class="prediction-probability" style="color:{status_color};">
                            {risk_probability:.1%}
                        </div>
                        <div class="prediction-description">
                            {recommendation}<br/>
                            <span style="font-size:0.85rem;">Model: {model_choice}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk_probability * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={
                            "text": "Cancer Risk Probability",
                            "font": {"color": "white", "size": 16},
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickcolor": "white",
                                "tickfont": {"color": "white"},
                            },
                            "bar": {
                                "color": "#dc2626" if pred_label == 1 else "#f87171"
                            },
                            "steps": [
                                {"range": [0, 30], "color": "rgba(52, 211, 153, 0.3)"},
                                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.3)"},
                                {"range": [60, 100], "color": "rgba(220, 38, 38, 0.3)"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Multi-model comparison for same patient
                st.caption("All-model probabilities for this patient")
                rows = []
                for name in pipeline.models:
                    _, probs = pipeline.predict(input_row, model_name=name)
                    rows.append(
                        {
                            "Model": name,
                            "Cancer Risk %": round(float(probs[0][1]) * 100, 2),
                            "Prediction": "High" if int(probs[0][1] > 0.5) else "Low",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(
            """
            <div class="modern-card">
                <h2>Model Performance Analytics</h2>
                <p style="color:#d4d4d4;">Scores from the latest training run on held-out test data</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                x=list(model_scores.keys()),
                y=list(model_scores.values()),
                title="Model Accuracy Comparison",
                labels={"x": "Models", "y": "Accuracy"},
                color=list(model_scores.values()),
                color_continuous_scale=["#b91c1c", "#f87171", "#fca5a5"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_color="#f87171",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Confusion Matrix — {best_model}",
                labels={"x": "Predicted", "y": "Actual"},
                color_continuous_scale=["#b91c1c", "#f87171", "#fca5a5"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_color="#f87171",
            )
            st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        cards = [
            (m1, metrics.get("accuracy", best_acc), "Accuracy"),
            (m2, metrics.get("precision", 0), "Precision"),
            (m3, metrics.get("recall", 0), "Recall"),
            (m4, metrics.get("f1", 0), "F1-Score"),
        ]
        for col, value, label in cards:
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{value:.0%}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        detail_rows = []
        for name, score in model_scores.items():
            mm = pipeline.model_metrics.get(name, {})
            detail_rows.append(
                {
                    "Model": name,
                    "Accuracy": round(score, 4),
                    "Precision": round(mm.get("precision", 0), 4),
                    "Recall": round(mm.get("recall", 0), 4),
                    "F1": round(mm.get("f1", 0), 4),
                }
            )
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    with tab3:
        if dataset is None:
            st.warning("cancer_dataset.csv not found.")
        else:
            country_data = (
                dataset.groupby("Country")
                .agg(Total_Cases=("Final_Prediction", "count"), Cancer_Cases=("Final_Prediction", "sum"))
                .reset_index()
            )
            country_data["Cancer_Rate"] = (
                country_data["Cancer_Cases"] / country_data["Total_Cases"] * 100
            ).round(1)

            fig = px.choropleth(
                country_data,
                locations="Country",
                locationmode="country names",
                color="Cancer_Rate",
                hover_name="Country",
                hover_data=["Total_Cases", "Cancer_Cases"],
                color_continuous_scale=["#fee2e2", "#f87171", "#b91c1c"],
                title="Cancer Rate by Country (training dataset)",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                geo_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_color="#f87171",
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    country_data.sort_values("Cancer_Rate", ascending=False),
                    x="Country",
                    y="Cancer_Rate",
                    title="Cancer Rate by Country",
                    color="Cancer_Rate",
                    color_continuous_scale=["#b91c1c", "#f87171", "#fca5a5"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    title_font_color="#f87171",
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.pie(
                    country_data,
                    values="Cancer_Cases",
                    names="Country",
                    title="Cancer Cases Distribution",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    title_font_color="#f87171",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if dataset is None:
            st.warning("cancer_dataset.csv not found.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    dataset,
                    x="Age",
                    color="Final_Prediction",
                    title="Age Distribution by Cancer Status",
                    nbins=20,
                    color_discrete_map={0: "#fca5a5", 1: "#dc2626"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    title_font_color="#f87171",
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                gender_cancer = (
                    dataset.groupby(["Gender", "Final_Prediction"])
                    .size()
                    .reset_index(name="Count")
                )
                fig = px.bar(
                    gender_cancer,
                    x="Gender",
                    y="Count",
                    color="Final_Prediction",
                    title="Cancer Cases by Gender",
                    color_discrete_map={0: "#fca5a5", 1: "#dc2626"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    title_font_color="#f87171",
                )
                st.plotly_chart(fig, use_container_width=True)

            smoking_cancer = (
                dataset.groupby(["Smoking_Status", "Final_Prediction"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                smoking_cancer,
                x="Smoking_Status",
                y="Count",
                color="Final_Prediction",
                title="Impact of Smoking on Cancer Risk",
                color_discrete_map={0: "#fca5a5", 1: "#dc2626"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_color="#f87171",
            )
            st.plotly_chart(fig, use_container_width=True)

            i1, i2, i3, i4 = st.columns(4)
            with i1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{len(dataset):,}</div>'
                    f'<div class="metric-label">Total Patients</div></div>',
                    unsafe_allow_html=True,
                )
            with i2:
                rate = dataset["Final_Prediction"].mean() * 100
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{rate:.1f}%</div>'
                    f'<div class="metric-label">Cancer Rate</div></div>',
                    unsafe_allow_html=True,
                )
            with i3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{dataset["Age"].mean():.0f}</div>'
                    f'<div class="metric-label">Average Age</div></div>',
                    unsafe_allow_html=True,
                )
            with i4:
                male_pct = (dataset["Gender"] == "Male").mean() * 100
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{male_pct:.0f}%</div>'
                    f'<div class="metric-label">Male Patients</div></div>',
                    unsafe_allow_html=True,
                )

    with tab5:
        rf = pipeline.models.get("Random Forest")
        if rf is not None and hasattr(rf, "feature_importances_"):
            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": pipeline.feature_names,
                        "Importance": rf.feature_importances_,
                    }
                )
                .sort_values("Importance", ascending=True)
                .tail(12)
            )
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (Random Forest)",
                color="Importance",
                color_continuous_scale=["#b91c1c", "#f87171", "#fca5a5"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_color="#f87171",
            )
            st.plotly_chart(fig, use_container_width=True)

            if dataset is not None:
                num_cols = [c for c in NUMERICAL_COLS if c in dataset.columns]
                corr = dataset[num_cols + ["Final_Prediction"]].corr()
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Numerical Feature Correlation",
                    color_continuous_scale=["#b91c1c", "#f87171", "#fee2e2"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    title_font_color="#f87171",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Random Forest feature importances unavailable.")

    st.markdown(
        f"""
        <div class="footer" style="text-align:center;">
            <div style="font-size:1.3rem;color:#f87171;font-weight:600;">MediCore Cancer Analytics Platform</div>
            <div style="color:#d4d4d4;">Live model inference · Developed by Umair Elahi</div>
            <div style="margin-top:1rem;color:#d4d4d4;font-size:0.9rem;">
                Best Accuracy: {best_acc:.1%} · Models: {len(pipeline.models)} · Features: {len(pipeline.feature_names)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

import csv
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from streamlit.errors import StreamlitSecretNotFoundError


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Career Progression & Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.markdown(
    """
    <style>
    :root {
        --ink: #172b4d;
        --muted: #64748b;
        --surface: #ffffff;
        --border: #e2e8f0;
        --accent: #2563eb;
    }
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 280px, #f7f9fc 100%);
        color: var(--ink);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #f8fbff 0%, #edf4ff 52%, #f8fafc 100%);
        border-right: 1px solid #dbe5f1;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink);
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--muted);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: var(--ink);
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #dbe5f1;
        border-radius: 10px;
        padding: 0.45rem 0.65rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
        background: #e8f0ff;
        border-color: #93b4f4;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] > div > div {
        background: var(--accent);
        border-color: var(--accent);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.78);
        border: 1px dashed #b8c9df;
        border-radius: 14px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent);
        background: #ffffff;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #eff6ff;
        border: 1px solid #c7d9ff;
        border-radius: 8px;
        color: #1e40af;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] .react-aria-ComboBox > [role="group"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #cbd8e8;
        border-radius: 12px;
        color: var(--ink);
        min-height: 2.75rem;
        box-shadow: 0 2px 8px rgba(30, 64, 110, 0.05);
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-tag] {
        background: #e8f0ff;
        border: 1px solid #c7d9ff;
        border-radius: 7px;
        color: #1e40af;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-tag] span {
        color: #1e40af;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-tag] button {
        color: #1e40af;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] input {
        color: var(--ink);
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] .react-aria-ComboBox > [role="group"]:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14);
    }
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] [role="listbox"] {
        background: #ffffff;
        color: var(--ink);
        border: 1px solid #dbe5f1;
        border-radius: 10px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.14);
    }
    [role="listbox"] {
        background: #ffffff !important;
        border: 1px solid #dbe5f1;
        border-radius: 10px;
        color: var(--ink) !important;
        padding: 0.75rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.14);
    }
    [role="listbox"] [role="option"],
    [role="listbox"] [role="option"] * {
        color: var(--ink) !important;
    }
    [data-baseweb="popover"] [role="option"] {
        color: var(--ink);
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background: #eff6ff;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
    }
    [data-testid="stPlotlyChart"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.5rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    employee_file = OUTPUT_DIR / "employee_career_scores.csv"
    cluster_file = OUTPUT_DIR / "career_cluster_summary.csv"
    evaluation_file = OUTPUT_DIR / "clustering_evaluation.csv"

    employees = pd.read_csv(employee_file)
    clusters = pd.read_csv(cluster_file)
    evaluation = pd.read_csv(evaluation_file)

    for column in ["Department", "JobRole"]:
        employees[column] = employees[column].astype("string")

    return employees, clusters, evaluation


@st.cache_data(show_spinner="Reading uploaded CSV...")
def load_uploaded_csv(contents):
    sample = contents[:65536].decode("utf-8-sig", errors="replace")
    try:
        delimiter = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|",
        ).delimiter
    except csv.Error:
        delimiter = ","

    try:
        return pd.read_csv(
            BytesIO(contents),
            sep=delimiter,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            BytesIO(contents),
            sep=delimiter,
            encoding="cp1252",
        )


@st.cache_data(show_spinner="Evaluating exploratory attrition baselines...")
def evaluate_attrition_models(data):
    excluded_columns = {
        "Attrition",
        "EmployeeRecordID",
        "Age",
        "Gender",
        "MaritalStatus",
        "PromotionGapRatio",
        "RoleStagnationIndex",
        "TrainingIntensityScore",
        "ManagerTenureRatio",
        "PromotionGapPoints",
        "RoleStagnationPoints",
        "TrainingPoints",
        "PromotionGapScore",
        "PromotionGapCategory",
        "LowSatisfactionIndicator",
        "LowTrainingIndicator",
        "RetentionOpportunityIndex",
        "InterventionPriority",
        "CareerCluster",
    }
    feature_columns = [
        column for column in data.columns if column not in excluded_columns
    ]
    features = data[feature_columns].copy()
    target_values = data["Attrition"].astype("string").str.strip().str.lower()
    target = target_values.isin({"1", "yes", "true", "left", "former"}).astype(int)

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    metric_rows = []
    importance_rows = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(train_features, train_target)
        predicted = pipeline.predict(test_features)
        probabilities = pipeline.predict_proba(test_features)[:, 1]
        cv_auc = cross_val_score(
            pipeline,
            features,
            target,
            cv=cross_validation,
            scoring="roc_auc",
            n_jobs=1,
        )
        metric_rows.append(
            {
                "Model": model_name,
                "Test accuracy": accuracy_score(test_target, predicted),
                "Test precision": precision_score(
                    test_target, predicted, zero_division=0
                ),
                "Test recall": recall_score(
                    test_target, predicted, zero_division=0
                ),
                "Test F1": f1_score(test_target, predicted, zero_division=0),
                "Test ROC-AUC": roc_auc_score(test_target, probabilities),
                "5-fold CV ROC-AUC": cv_auc.mean(),
                "CV ROC-AUC std": cv_auc.std(),
            }
        )

        transformed_names = pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        classifier = pipeline.named_steps["classifier"]
        if model_name == "Logistic Regression":
            raw_importance = abs(classifier.coef_[0])
        else:
            raw_importance = classifier.feature_importances_

        grouped_importance = {}
        for name, value in zip(transformed_names, raw_importance):
            feature_name = name.split("__", 1)[-1].split("_", 1)[0]
            grouped_importance[feature_name] = (
                grouped_importance.get(feature_name, 0.0) + float(value)
            )
        for feature_name, value in grouped_importance.items():
            importance_rows.append(
                {
                    "Model": model_name,
                    "Feature": feature_name,
                    "Importance": value,
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def style_chart(figure, height=380):
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 16, "r": 16, "t": 56, "b": 16},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#334155"},
        title={"font": {"size": 16, "color": "#172b4d"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"bgcolor": "#172b4d", "font_color": "#ffffff"},
    )
    return figure


def render_csv_analysis(data, filename):
    st.subheader(f"CSV analysis · {filename}")
    st.caption(
        "General-purpose analysis for any tabular CSV. "
        "The uploaded file is analyzed in memory and is not saved to project files."
    )

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in data.columns
        if column not in numeric_columns
    ]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(data):,}")
    metric_columns[1].metric("Columns", f"{len(data.columns):,}")
    metric_columns[2].metric("Numeric fields", f"{len(numeric_columns):,}")
    metric_columns[3].metric("Duplicate rows", f"{data.duplicated().sum():,}")

    missing = data.isna().sum().rename("Missing values").to_frame()
    missing["Missing %"] = (missing["Missing values"] / len(data) * 100).round(2)
    missing = missing.sort_values(
        ["Missing values", "Missing %"],
        ascending=False,
    )

    with st.expander("Data quality and column profile", expanded=False):
        profile = pd.DataFrame({
            "Column": data.columns,
            "Data type": data.dtypes.astype(str).to_numpy(),
            "Missing values": data.isna().sum().to_numpy(),
            "Unique values": data.nunique(dropna=True).to_numpy(),
        })
        left, right = st.columns(2)
        with left:
            st.markdown("**Column profile**")
            st.dataframe(profile, width="stretch", hide_index=True)
        with right:
            st.markdown("**Missing values**")
            st.dataframe(missing, width="stretch")

    if numeric_columns:
        st.subheader("Numeric overview")
        st.dataframe(
            data[numeric_columns].describe().T.round(2),
            width="stretch",
        )
        selected_numeric = st.selectbox(
            "Numeric field to visualize",
            options=numeric_columns,
        )
        numeric_values = data[selected_numeric].dropna()
        if not numeric_values.empty:
            figure = px.histogram(
                numeric_values,
                x=selected_numeric,
                nbins=min(40, max(10, int(numeric_values.nunique()))),
                title=f"Distribution of {selected_numeric}",
                labels={selected_numeric: selected_numeric},
                color_discrete_sequence=["#2563eb"],
            )
            style_chart(figure)
            st.plotly_chart(
                figure,
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.info(f"{selected_numeric} has no non-missing values to plot.")
    else:
        st.info("No numeric fields were detected; column profile and preview are available.")

    if categorical_columns:
        st.subheader("Category breakdown")
        selected_category = st.selectbox(
            "Categorical field to visualize",
            options=categorical_columns,
        )
        category_counts = (
            data[selected_category]
            .fillna("(Missing)")
            .astype(str)
            .value_counts()
            .head(15)
            .rename_axis(selected_category)
            .reset_index(name="Row count")
        )
        figure = px.bar(
            category_counts.sort_values("Row count"),
            x="Row count",
            y=selected_category,
            orientation="h",
            title=f"Top values in {selected_category}",
            text="Row count",
            color_discrete_sequence=["#2563eb"],
        )
        style_chart(figure, height=max(320, 34 * len(category_counts) + 100))
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displayModeBar": False},
        )

    st.subheader("Data preview")
    st.dataframe(data.head(500), width="stretch", hide_index=True)
    st.caption("Preview is limited to the first 500 rows.")

    st.download_button(
        "Download analyzed CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"analyzed_{Path(filename).name}",
        mime="text/csv",
    )


header, account = st.columns([4, 1], vertical_alignment="center")

with header:
    st.title("📊 Career Progression & Retention Analytics")
    st.markdown(
        """
        **Explore employee career progression, promotion gaps, workforce
        segments, and any CSV dataset.**
        """
    )

with account:
    if st.user.get("is_logged_in", False):
        account_name = st.user.get("name") or st.user.get("email") or "Signed in"
        st.caption(f"👤 {account_name}")
        if st.button("Log out", width="stretch"):
            st.logout()
    else:
        try:
            auth_settings = st.secrets.get("auth", {})
        except StreamlitSecretNotFoundError:
            auth_settings = {}
        google_ready = all(
            auth_settings.get(key)
            for key in (
                "redirect_uri",
                "cookie_secret",
                "client_id",
                "client_secret",
                "server_metadata_url",
            )
        )
        if google_ready:
            if st.button("Sign in with Google", width="stretch"):
                st.login()
        else:
            st.button("Sign in with Google", width="stretch", disabled=True)
            st.caption("Configure Google OAuth to enable sign-in.")

st.caption(
    "Career indicators are exploratory, rule-based measures—not validated "
    "predictions of employee resignation. Sign-in is optional."
)

st.sidebar.header("Analyze a CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="Upload comma-, semicolon-, tab-, or pipe-delimited CSV data.",
)

if uploaded_file is not None:
    try:
        uploaded_data = load_uploaded_csv(uploaded_file.getvalue())
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

    if uploaded_data.empty or len(uploaded_data.columns) == 0:
        st.error("The uploaded CSV does not contain any data rows or columns.")
        st.stop()

    render_csv_analysis(uploaded_data, uploaded_file.name)
    st.stop()


# --------------------------------------------------
# ERROR HANDLING
# --------------------------------------------------

try:
    employees, clusters, evaluation = load_data()
except FileNotFoundError as error:
    st.error(
        f"Required output file not found: {error.filename}. "
        "Run src/career_model.py first."
    )
    st.stop()


required_columns = [
    "Department",
    "JobRole",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "TrainingTimesLastYear",
    "JobSatisfaction",
    "PromotionGapCategory",
    "InterventionPriority",
    "CareerCluster",
    "Attrition",
]

missing_columns = [
    column for column in required_columns
    if column not in employees.columns
]

if missing_columns:
    st.error(f"Missing columns in employee data: {missing_columns}")
    st.stop()


# --------------------------------------------------
# NAVIGATION AND SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.subheader("Dashboard")
selected_page = st.sidebar.radio(
    "Navigate to",
    options=[
        "Executive Overview",
        "Promotion Gap Analysis",
        "Retention Intelligence",
        "Insights & Recommendations",
        "Full Analysis",
    ],
    label_visibility="collapsed",
)

st.sidebar.header("Dashboard Filters")
st.sidebar.caption("Filters apply across every chart and employee record.")

departments = sorted(
    employees["Department"].dropna().unique()
)

selected_departments = st.sidebar.multiselect(
    "Select Department",
    options=departments,
    default=departments
)

roles = sorted(
    employees["JobRole"].dropna().unique()
)

selected_roles = st.sidebar.multiselect(
    "Select Job Role",
    options=roles,
    default=roles
)

risk_categories = ["High", "Medium", "Low"]

selected_risk = st.sidebar.multiselect(
    "Promotion Gap Category",
    options=risk_categories,
    default=risk_categories
)

priorities = ["High", "Medium", "Low"]

selected_priority = st.sidebar.multiselect(
    "Intervention Priority",
    options=priorities,
    default=priorities
)


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered = employees[
    employees["Department"].isin(selected_departments)
    & employees["JobRole"].isin(selected_roles)
    & employees["PromotionGapCategory"].isin(selected_risk)
    & employees["InterventionPriority"].isin(selected_priority)
].copy()


if filtered.empty:
    st.warning(
        "No employees match the selected filters. "
        "Please change your filter selections."
    )
    st.stop()


# --------------------------------------------------
# ADVANCED PROMOTION GAP ANALYSIS
# --------------------------------------------------

if selected_page == "Promotion Gap Analysis":
    st.subheader("Promotion Gap Analysis")
    st.caption(
        "Compare current promotion-gap indicators across the selected workforce. "
        "These measures do not establish promotion eligibility or unfair treatment."
    )

    gap_metrics = st.columns(4)
    gap_metrics[0].metric(
        "Average Years Since Promotion",
        f"{filtered['YearsSinceLastPromotion'].mean():.2f}",
    )
    gap_metrics[1].metric(
        "Median Years Since Promotion",
        f"{filtered['YearsSinceLastPromotion'].median():.2f}",
    )
    gap_metrics[2].metric(
        "Average Years in Current Role",
        f"{filtered['YearsInCurrentRole'].mean():.2f}",
    )
    high_gap = int((filtered["PromotionGapCategory"] == "High").sum())
    gap_metrics[3].metric(
        "High Gap Category",
        f"{high_gap:,}",
        delta=f"{high_gap / len(filtered):.1%} of view",
        delta_color="inverse",
    )

    department_gap = (
        filtered.groupby("Department", as_index=False)
        .agg(
            Employees=("Department", "size"),
            AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
            MedianPromotionGap=("YearsSinceLastPromotion", "median"),
            AverageRoleTenure=("YearsInCurrentRole", "mean"),
        )
    )
    role_gap = (
        filtered.groupby("JobRole", as_index=False)
        .agg(
            Employees=("JobRole", "size"),
            AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
            MedianPromotionGap=("YearsSinceLastPromotion", "median"),
            AverageRoleTenure=("YearsInCurrentRole", "mean"),
        )
        .sort_values("AveragePromotionGap", ascending=False)
    )

    gap_charts = st.columns(2)
    department_figure = px.bar(
        department_gap.sort_values("AveragePromotionGap"),
        x="AveragePromotionGap",
        y="Department",
        orientation="h",
        color="AveragePromotionGap",
        color_continuous_scale="Blues",
        text_auto=".2f",
        hover_data=["Employees", "MedianPromotionGap", "AverageRoleTenure"],
        title="Promotion Gap by Department",
        labels={"AveragePromotionGap": "Average years"},
    )
    style_chart(department_figure).update_layout(
        yaxis={"categoryorder": "total ascending"}
    )
    gap_charts[0].plotly_chart(
        department_figure,
        width="stretch",
        config={"displayModeBar": False},
    )

    role_figure = px.bar(
        role_gap,
        x="AveragePromotionGap",
        y="JobRole",
        orientation="h",
        color="AveragePromotionGap",
        color_continuous_scale="Blues",
        text_auto=".2f",
        hover_data=["Employees", "MedianPromotionGap", "AverageRoleTenure"],
        title="Promotion Gap by Job Role",
        labels={"AveragePromotionGap": "Average years"},
    )
    style_chart(role_figure, height=max(380, 34 * len(role_gap) + 100)).update_layout(
        yaxis={"categoryorder": "total ascending"}
    )
    gap_charts[1].plotly_chart(
        role_figure,
        width="stretch",
        config={"displayModeBar": False},
    )

    gap_scatter = px.scatter(
        filtered,
        x="YearsAtCompany",
        y="YearsSinceLastPromotion",
        color="PromotionGapCategory",
        size="YearsInCurrentRole",
        hover_data=[
            column for column in ["Department", "JobRole", "JobLevel"]
            if column in filtered.columns
        ],
        color_discrete_map={
            "Low": "#10b981",
            "Medium": "#f59e0b",
            "High": "#ef4444",
        },
        title="Company Tenure vs. Years Since Last Promotion",
        labels={
            "YearsAtCompany": "Years at company",
            "YearsSinceLastPromotion": "Years since last promotion",
        },
    )
    style_chart(gap_scatter)
    st.plotly_chart(
        gap_scatter,
        width="stretch",
        config={"displayModeBar": False},
    )

    if "JobLevel" in filtered.columns:
        level_gap = (
            filtered.groupby("JobLevel", as_index=False)
            .agg(
                Employees=("JobLevel", "size"),
                AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
            )
        )
        level_figure = px.bar(
            level_gap,
            x="JobLevel",
            y="AveragePromotionGap",
            color="AveragePromotionGap",
            color_continuous_scale="Blues",
            text_auto=".2f",
            hover_data=["Employees"],
            title="Promotion Gap by Job Level",
            labels={"AveragePromotionGap": "Average years"},
        )
        style_chart(level_figure).update_layout(showlegend=False)
        st.plotly_chart(
            level_figure,
            width="stretch",
            config={"displayModeBar": False},
        )

    st.stop()


# --------------------------------------------------
# RETENTION INTELLIGENCE
# --------------------------------------------------

if selected_page == "Retention Intelligence":
    st.subheader("Retention Intelligence")
    st.caption(
        "Historical attrition patterns and exploratory model benchmarks. "
        "This page does not score individual employees or predict specific exits."
    )

    if "Attrition" not in filtered.columns:
        st.info("Attrition data is unavailable in the selected dataset.")
        st.stop()

    attrition_labels = filtered["Attrition"].astype("string").str.strip().str.lower()
    former_labels = {"1", "yes", "true", "left", "former"}
    retained_labels = {"0", "no", "false", "retained", "active"}
    former_mask = attrition_labels.isin(former_labels)
    known_mask = former_mask | attrition_labels.isin(retained_labels)
    attrition_data = filtered.loc[known_mask].copy()

    if attrition_data.empty:
        st.warning("No recognized retained/former employee values were found.")
        st.stop()

    attrition_data["Former employee"] = former_mask.loc[known_mask].astype(int)
    former_count = int(attrition_data["Former employee"].sum())
    total_known = len(attrition_data)
    retained_count = total_known - former_count
    retention_metrics = st.columns(4)
    retention_metrics[0].metric("Employees in View", f"{total_known:,}")
    retention_metrics[1].metric("Retained Employees", f"{retained_count:,}")
    retention_metrics[2].metric("Former Employees", f"{former_count:,}")
    retention_metrics[3].metric(
        "Observed Attrition",
        f"{former_count / total_known:.1%}",
    )

    st.markdown("#### Observed group patterns")

    def attrition_by(column):
        summary = (
            attrition_data.groupby(column, dropna=False, as_index=False)
            .agg(
                Employees=(column, "size"),
                FormerEmployees=("Former employee", "sum"),
            )
        )
        summary["Observed attrition"] = (
            summary["FormerEmployees"] / summary["Employees"]
        )
        return summary

    department_attrition = attrition_by("Department")
    role_attrition = attrition_by("JobRole")
    retention_charts = st.columns(2)
    for container, summary, dimension, title in [
        (
            retention_charts[0],
            department_attrition,
            "Department",
            "Observed Attrition by Department",
        ),
        (
            retention_charts[1],
            role_attrition,
            "JobRole",
            "Observed Attrition by Job Role",
        ),
    ]:
        chart = px.bar(
            summary.sort_values("Observed attrition"),
            x="Observed attrition",
            y=dimension,
            orientation="h",
            color="Observed attrition",
            color_continuous_scale="Oranges",
            text_auto=".1%",
            hover_data=["Employees", "FormerEmployees"],
            title=title,
        )
        style_chart(chart, height=max(380, 34 * len(summary) + 100)).update_layout(
            xaxis_tickformat=".0%",
            yaxis={"categoryorder": "total ascending"},
        )
        container.plotly_chart(
            chart,
            width="stretch",
            config={"displayModeBar": False},
        )

    factor_charts = st.columns(2)
    if "OverTime" in attrition_data.columns:
        overtime_attrition = attrition_by("OverTime")
        overtime_chart = px.bar(
            overtime_attrition,
            x="OverTime",
            y="Observed attrition",
            color="OverTime",
            text_auto=".1%",
            hover_data=["Employees", "FormerEmployees"],
            title="Observed Attrition by Overtime",
        )
        style_chart(overtime_chart).update_layout(
            yaxis_tickformat=".0%",
            showlegend=False,
        )
        factor_charts[0].plotly_chart(
            overtime_chart,
            width="stretch",
            config={"displayModeBar": False},
        )
    if "JobSatisfaction" in attrition_data.columns:
        satisfaction_attrition = attrition_by("JobSatisfaction")
        satisfaction_chart = px.bar(
            satisfaction_attrition,
            x="JobSatisfaction",
            y="Observed attrition",
            color="Observed attrition",
            color_continuous_scale="Oranges",
            text_auto=".1%",
            hover_data=["Employees", "FormerEmployees"],
            title="Observed Attrition by Job Satisfaction",
        )
        style_chart(satisfaction_chart).update_layout(yaxis_tickformat=".0%")
        factor_charts[1].plotly_chart(
            satisfaction_chart,
            width="stretch",
            config={"displayModeBar": False},
        )

    st.markdown("#### Exploratory model benchmark")
    st.caption(
        "Models use a stratified 75/25 train-test split and 5-fold "
        "cross-validation. Age, gender, marital status, employee identifiers, "
        "and derived career-risk scores are excluded. Other inputs may still "
        "act as proxies; these retrospective results are not validated for "
        "future prediction or employment decisions."
    )
    attrition_values = (
        employees["Attrition"].astype("string").str.strip().str.lower()
    )
    recognized_labels = former_labels | retained_labels
    model_data = employees.loc[attrition_values.isin(recognized_labels)]
    model_class_counts = attrition_values.loc[
        attrition_values.isin(recognized_labels)
    ].map(lambda value: int(value in former_labels)).value_counts()
    if len(model_data) < 50 or model_class_counts.min() < 10:
        st.info(
            "At least 50 labeled employee records and 10 records in each "
            "outcome group are required for this exploratory benchmark."
        )
    else:
        model_data = model_data.copy()
        model_data["Attrition"] = (
            model_data["Attrition"].astype("string").str.strip().str.lower()
            .isin(former_labels).astype(int)
        )
        try:
            model_metrics, model_importance = evaluate_attrition_models(model_data)
        except ValueError as error:
            st.error(f"Could not evaluate attrition models: {error}")
            st.stop()

        metric_display = model_metrics.copy()
        metric_columns = [
            column for column in metric_display.columns if column != "Model"
        ]
        metric_display[metric_columns] = metric_display[metric_columns].round(3)
        st.dataframe(metric_display, width="stretch", hide_index=True)

        importance_display = (
            model_importance.groupby("Feature", as_index=False)["Importance"]
            .mean()
            .sort_values("Importance", ascending=False)
            .head(10)
        )
        importance_chart = px.bar(
            importance_display.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Top Global Model Features (Average Importance)",
        )
        style_chart(importance_chart, height=420).update_layout(
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(
            importance_chart,
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            "Feature importance describes model behavior in this dataset; "
            "it does not show causation or explain an individual outcome."
        )

    st.stop()


# --------------------------------------------------
# AUTOMATED MANAGEMENT INSIGHTS
# --------------------------------------------------

if selected_page == "Insights & Recommendations":
    st.subheader("Insights & Recommendations")
    st.caption(
        "Evidence-linked observations for the current filters. "
        "Recommendations are prompts for review, not automated decisions."
    )

    department_insights = (
        filtered.groupby("Department", as_index=False)
        .agg(
            Employees=("Department", "size"),
            AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
            AverageRoleTenure=("YearsInCurrentRole", "mean"),
            AverageTraining=("TrainingTimesLastYear", "mean"),
        )
    )
    role_insights = (
        filtered.groupby("JobRole", as_index=False)
        .agg(
            Employees=("JobRole", "size"),
            AverageRoleTenure=("YearsInCurrentRole", "mean"),
            AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        )
    )

    insight_count = st.columns(3)
    insight_count[0].metric("Employees in View", f"{len(filtered):,}")
    insight_count[1].metric(
        "Departments Compared",
        f"{len(department_insights):,}",
    )
    insight_count[2].metric("Job Roles Compared", f"{len(role_insights):,}")

    highest_gap = department_insights.sort_values(
        "AveragePromotionGap", ascending=False
    ).iloc[0]
    lowest_training = department_insights.sort_values(
        "AverageTraining", ascending=True
    ).iloc[0]
    longest_role = role_insights.sort_values(
        "AverageRoleTenure", ascending=False
    ).iloc[0]

    st.markdown("#### Findings to review")
    st.info(
        f"**Promotion gap:** {highest_gap['Department']} has the highest "
        f"department average in this view at "
        f"{highest_gap['AveragePromotionGap']:.2f} years "
        f"(n={int(highest_gap['Employees']):,}). Review role mix, career "
        "path access, and promotion criteria before drawing conclusions."
    )
    st.info(
        f"**Role tenure:** {longest_role['JobRole']} has the highest average "
        f"time in current role at {longest_role['AverageRoleTenure']:.2f} "
        f"years (n={int(longest_role['Employees']):,}). Consider reviewing "
        "development and internal mobility options with relevant teams."
    )
    st.info(
        f"**Training:** {lowest_training['Department']} has the lowest "
        f"average training frequency in this view at "
        f"{lowest_training['AverageTraining']:.2f} sessions "
        f"(n={int(lowest_training['Employees']):,}). Check training access "
        "and role-specific needs."
    )

    if "Attrition" in filtered.columns:
        insight_attrition_labels = (
            filtered["Attrition"].astype("string").str.strip().str.lower()
        )
        insight_former = insight_attrition_labels.isin(
            {"1", "yes", "true", "left", "former"}
        )
        insight_retained = insight_attrition_labels.isin(
            {"0", "no", "false", "retained", "active"}
        )
        insight_known = insight_former | insight_retained
        if insight_known.any():
            attrition_insights = filtered.loc[insight_known].copy()
            attrition_insights["Former employee"] = insight_former.loc[
                insight_known
            ].astype(int)
            attrition_by_department = (
                attrition_insights.groupby("Department", as_index=False)
                .agg(
                    Employees=("Department", "size"),
                    FormerEmployees=("Former employee", "sum"),
                )
            )
            attrition_by_department = attrition_by_department[
                attrition_by_department["Employees"] >= 10
            ]
            if not attrition_by_department.empty:
                attrition_by_department["Observed attrition"] = (
                    attrition_by_department["FormerEmployees"]
                    / attrition_by_department["Employees"]
                )
                overall_rate = (
                    attrition_insights["Former employee"].mean()
                )
                higher_attrition = attrition_by_department.sort_values(
                    "Observed attrition", ascending=False
                ).iloc[0]
                if higher_attrition["Observed attrition"] > overall_rate:
                    st.info(
                        f"**Observed attrition:** {higher_attrition['Department']} "
                        f"is {higher_attrition['Observed attrition']:.1%} "
                        f"(n={int(higher_attrition['Employees']):,}), versus "
                        f"{overall_rate:.1%} across the filtered view. "
                        "Review group context and data limitations before "
                        "considering follow-up."
                    )

    st.markdown("#### Department comparison")
    department_display = department_insights.copy()
    numeric_department_columns = [
        column for column in department_display.columns
        if column.startswith("Average")
    ]
    department_display[numeric_department_columns] = department_display[
        numeric_department_columns
    ].round(2)
    st.dataframe(department_display, width="stretch", hide_index=True)

    st.caption(
        "All observations are descriptive associations. They do not establish "
        "causes, fairness, promotion eligibility, or an employee's likelihood "
        "of leaving."
    )
    st.stop()


# --------------------------------------------------
# EXECUTIVE OVERVIEW
# --------------------------------------------------

if selected_page == "Executive Overview":
    st.subheader("Executive Overview")
    st.caption(
        f"At-a-glance career and retention indicators for "
        f"{len(filtered):,} employees matching the selected filters."
    )

    overview_metrics = st.columns(5)
    overview_metrics[0].metric("Employees in View", f"{len(filtered):,}")

    if "Attrition" in filtered.columns:
        attrition_values = (
            filtered["Attrition"].astype("string").str.strip().str.lower()
        )
        former_values = {"1", "yes", "true", "left", "former"}
        retained_values = {"0", "no", "false", "retained", "active"}
        former_mask = attrition_values.isin(former_values)
        known_attrition = former_mask | attrition_values.isin(retained_values)
        former_count = int(former_mask.sum())

        if known_attrition.any():
            attrition_rate = former_count / int(known_attrition.sum())
            overview_metrics[1].metric(
                "Former Employees",
                f"{former_count:,}",
                delta=f"{attrition_rate:.1%} observed attrition",
                delta_color="inverse",
            )
        else:
            overview_metrics[1].metric("Former Employees", "N/A")
    else:
        attrition_values = None
        former_mask = None
        known_attrition = None
        overview_metrics[1].metric("Former Employees", "Not available")

    overview_metrics[2].metric(
        "Avg. Years Since Promotion",
        f"{filtered['YearsSinceLastPromotion'].mean():.2f}",
    )
    high_gap_count = int((filtered["PromotionGapCategory"] == "High").sum())
    overview_metrics[3].metric(
        "High Promotion Gap",
        f"{high_gap_count:,}",
        delta=f"{high_gap_count / len(filtered):.1%} of view",
        delta_color="inverse",
    )
    high_priority_count = int(
        (filtered["InterventionPriority"] == "High").sum()
    )
    overview_metrics[4].metric(
        "High Intervention Priority",
        f"{high_priority_count:,}",
        delta=f"{high_priority_count / len(filtered):.1%} of view",
        delta_color="inverse",
    )

    st.markdown("#### Highlights to investigate")
    highlight_columns = st.columns(2)
    department_highlights = (
        filtered.groupby("Department", as_index=False)
        .agg(
            Employees=("Department", "size"),
            AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        )
        .sort_values("AveragePromotionGap", ascending=False)
    )
    if not department_highlights.empty:
        leading_department = department_highlights.iloc[0]
        highlight_columns[0].info(
            f"**Highest department average promotion gap:** "
            f"{leading_department['Department']} — "
            f"{leading_department['AveragePromotionGap']:.2f} years "
            f"({int(leading_department['Employees']):,} employees)."
        )

    role_highlights = (
        filtered.groupby("JobRole", as_index=False)
        .agg(
            Employees=("JobRole", "size"),
            AverageRoleTenure=("YearsInCurrentRole", "mean"),
        )
        .sort_values("AverageRoleTenure", ascending=False)
    )
    if not role_highlights.empty:
        leading_role = role_highlights.iloc[0]
        highlight_columns[1].info(
            f"**Longest average time in current role:** "
            f"{leading_role['JobRole']} — "
            f"{leading_role['AverageRoleTenure']:.2f} years "
            f"({int(leading_role['Employees']):,} employees)."
        )
    st.caption(
        "These are descriptive comparisons, not evidence of causes, "
        "unfair treatment, or individual resignation risk."
    )

    overview_charts = st.columns(2)
    risk_counts = (
        filtered["PromotionGapCategory"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Promotion Gap Category")
        .reset_index(name="Employee Count")
    )
    overview_risk = px.bar(
        risk_counts,
        x="Promotion Gap Category",
        y="Employee Count",
        color="Promotion Gap Category",
        category_orders={"Promotion Gap Category": ["Low", "Medium", "High"]},
        color_discrete_map={
            "Low": "#10b981",
            "Medium": "#f59e0b",
            "High": "#ef4444",
        },
        title="Promotion Gap Categories",
        text="Employee Count",
    )
    style_chart(overview_risk).update_layout(showlegend=False)
    overview_charts[0].plotly_chart(
        overview_risk,
        width="stretch",
        config={"displayModeBar": False},
    )

    overview_department = px.bar(
        department_highlights.sort_values("AveragePromotionGap"),
        x="AveragePromotionGap",
        y="Department",
        orientation="h",
        color="AveragePromotionGap",
        color_continuous_scale="Blues",
        text_auto=".2f",
        hover_data=["Employees"],
        title="Average Promotion Gap by Department",
        labels={"AveragePromotionGap": "Average years"},
    )
    style_chart(overview_department).update_layout(
        yaxis={"categoryorder": "total ascending"}
    )
    overview_charts[1].plotly_chart(
        overview_department,
        width="stretch",
        config={"displayModeBar": False},
    )

    if (
        attrition_values is not None
        and former_mask is not None
        and known_attrition is not None
        and known_attrition.any()
    ):
        attrition_data = filtered.loc[known_attrition].copy()
        attrition_data["Former employee"] = former_mask.loc[known_attrition]
        department_attrition = (
            attrition_data.groupby("Department", as_index=False)
            .agg(
                Employees=("Department", "size"),
                FormerEmployees=("Former employee", "sum"),
            )
        )
        department_attrition["Observed attrition"] = (
            department_attrition["FormerEmployees"]
            / department_attrition["Employees"]
        )
        overview_attrition = px.bar(
            department_attrition.sort_values("Observed attrition"),
            x="Observed attrition",
            y="Department",
            orientation="h",
            color="Observed attrition",
            color_continuous_scale="Oranges",
            text_auto=".1%",
            hover_data=["Employees", "FormerEmployees"],
            title="Observed Attrition by Department",
        )
        style_chart(overview_attrition).update_layout(
            xaxis_tickformat=".0%",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(
            overview_attrition,
            width="stretch",
            config={"displayModeBar": False},
        )

    st.stop()


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

total_employees = len(filtered)

high_gap_count = (
    filtered["PromotionGapCategory"] == "High"
).sum()

high_priority_count = (
    filtered["InterventionPriority"] == "High"
).sum()

average_promotion_gap = filtered[
    "YearsSinceLastPromotion"
].mean()

average_role_tenure = filtered[
    "YearsInCurrentRole"
].mean()


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Employees in View",
    f"{total_employees:,}"
)

col2.metric(
    "High Promotion Gap",
    f"{high_gap_count:,}",
    delta=f"{high_gap_count / total_employees:.1%} of view",
    delta_color="inverse",
)

col3.metric(
    "High Intervention Priority",
    f"{high_priority_count:,}",
    delta=f"{high_priority_count / total_employees:.1%} of view",
    delta_color="inverse",
)

col4.metric(
    "Avg. Years Since Promotion",
    f"{average_promotion_gap:.2f}"
)

col5.metric(
    "Avg. Years in Current Role",
    f"{average_role_tenure:.2f}"
)


st.divider()


# --------------------------------------------------
# CHART 1: PROMOTION GAP CATEGORY
# --------------------------------------------------

st.subheader("1. Promotion Gap Analysis")

risk_order = ["Low", "Medium", "High"]

risk_counts = (
    filtered["PromotionGapCategory"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
    .rename_axis("Promotion Gap Category")
    .reset_index(name="Employee Count")
)

fig_risk = px.bar(
    risk_counts,
    x="Promotion Gap Category",
    y="Employee Count",
    color="Promotion Gap Category",
    category_orders={
        "Promotion Gap Category": risk_order
    },
    title="Employees by Promotion Gap Category",
    text="Employee Count",
    color_discrete_map={
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444",
    },
)

style_chart(fig_risk).update_layout(showlegend=False)

st.plotly_chart(
    fig_risk,
    width="stretch",
    config={"displayModeBar": False},
)


# --------------------------------------------------
# CHART 2: DEPARTMENT COMPARISON
# --------------------------------------------------

st.subheader("2. Department-wise Career Progression")

department_summary = (
    filtered.groupby("Department", as_index=False)
    .agg(
        EmployeeCount=("Department", "size"),
        AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        AverageRoleTenure=("YearsInCurrentRole", "mean"),
        AverageTraining=("TrainingTimesLastYear", "mean"),
    )
)

fig_department = px.bar(
    department_summary,
    x="Department",
    y="AveragePromotionGap",
    color="Department",
    text_auto=".2f",
    title="Average Years Since Last Promotion by Department",
    hover_data=[
        "EmployeeCount",
        "AverageRoleTenure",
        "AverageTraining",
    ],
    color_discrete_sequence=["#2563eb"],
)

style_chart(fig_department).update_layout(showlegend=False)

st.plotly_chart(
    fig_department,
    width="stretch",
    config={"displayModeBar": False},
)


# --------------------------------------------------
# CHART 3: JOB ROLE COMPARISON
# --------------------------------------------------

st.subheader("3. Job Role Analysis")

role_summary = (
    filtered.groupby("JobRole", as_index=False)
    .agg(
        EmployeeCount=("JobRole", "size"),
        AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        AverageRoleTenure=("YearsInCurrentRole", "mean"),
    )
    .sort_values(
        "AveragePromotionGap",
        ascending=False
    )
)

fig_roles = px.bar(
    role_summary,
    x="AveragePromotionGap",
    y="JobRole",
    orientation="h",
    color="AveragePromotionGap",
    text_auto=".2f",
    title="Average Promotion Gap by Job Role",
    hover_data=[
        "EmployeeCount",
        "AverageRoleTenure",
    ],
    color_continuous_scale="Blues",
)

style_chart(fig_roles).update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    fig_roles,
    width="stretch",
    config={"displayModeBar": False},
)


# --------------------------------------------------
# CHART 4: INTERVENTION PRIORITY
# --------------------------------------------------

st.subheader("4. Retention Intervention Priorities")

priority_counts = (
    filtered["InterventionPriority"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
    .rename_axis("Intervention Priority")
    .reset_index(name="Employee Count")
)

fig_priority = px.pie(
    priority_counts,
    names="Intervention Priority",
    values="Employee Count",
    title="Distribution of Intervention Priority",
    hole=0.45,
    color_discrete_map={
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#10b981",
    },
)

style_chart(fig_priority)
st.plotly_chart(
    fig_priority,
    width="stretch",
    config={"displayModeBar": False},
)


# --------------------------------------------------
# CHART 5: CAREER CLUSTERS
# --------------------------------------------------

st.subheader("5. Employee Career Segmentation")

cluster_display = clusters.copy()

if "CareerCluster" in cluster_display.columns:
    cluster_display["CareerCluster"] = (
        cluster_display["CareerCluster"].astype(str)
    )

cluster_columns = [
    column for column in [
        "CareerCluster",
        "EmployeeCount",
        "AverageCompanyTenure",
        "AverageRoleTenure",
        "AveragePromotionGap",
        "AverageTraining",
        "AverageJobSatisfaction",
        "HighGapEmployees",
    ]
    if column in cluster_display.columns
]

st.dataframe(
    cluster_display[cluster_columns],
    width="stretch",
    hide_index=True
)

if {
    "CareerCluster",
    "AveragePromotionGap",
    "AverageRoleTenure",
}.issubset(cluster_display.columns):

    fig_clusters = px.scatter(
        cluster_display,
        x="AverageRoleTenure",
        y="AveragePromotionGap",
        size="EmployeeCount",
        color="CareerCluster",
        hover_data=[
            column for column in [
                "EmployeeCount",
                "AverageCompanyTenure",
                "AverageTraining",
                "AverageJobSatisfaction",
            ]
            if column in cluster_display.columns
        ],
        title="Career Clusters: Role Tenure vs Promotion Gap",
    )

    style_chart(fig_clusters)
    st.plotly_chart(
        fig_clusters,
        width="stretch",
        config={"displayModeBar": False},
    )


# --------------------------------------------------
# CLUSTER EVALUATION
# --------------------------------------------------

st.subheader("6. Clustering Model Evaluation")

if {"k", "silhouette_score"}.issubset(evaluation.columns):

    fig_evaluation = px.line(
        evaluation,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette Score for Different Cluster Counts",
        labels={
            "k": "Number of Clusters",
            "silhouette_score": "Silhouette Score",
        },
    )

    style_chart(fig_evaluation)
    st.plotly_chart(
        fig_evaluation,
        width="stretch",
        config={"displayModeBar": False},
    )

    st.caption(
        "A higher silhouette score indicates better separation and "
        "cohesion among clusters for the tested configurations. "
        "It does not establish that the clusters are business-optimal."
    )

else:
    st.info(
        "Clustering evaluation columns were not found. "
        "Check outputs/clustering_evaluation.csv."
    )


# --------------------------------------------------
# EMPLOYEE-LEVEL EXPLORATION
# --------------------------------------------------

st.subheader("7. Employee Career Records")

search_text = st.text_input(
    "Search by Department or Job Role"
)

employee_view = filtered.copy()

if search_text:
    search_mask = (
        employee_view["Department"].str.contains(
            search_text,
            case=False,
            na=False,
            regex=False,
        )
        |
        employee_view["JobRole"].str.contains(
            search_text,
            case=False,
            na=False,
            regex=False,
        )
    )

    employee_view = employee_view[search_mask]


display_columns = [
    column for column in [
        "EmployeeRecordID",
        "Department",
        "JobRole",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "TrainingTimesLastYear",
        "JobSatisfaction",
        "PromotionGapScore",
        "PromotionGapCategory",
        "RetentionOpportunityIndex",
        "InterventionPriority",
        "CareerCluster",
    ]
    if column in employee_view.columns
]

st.dataframe(
    employee_view[display_columns],
    width="stretch",
    hide_index=True
)

csv_data = employee_view.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Employee Data",
    data=csv_data,
    file_name="filtered_career_analysis.csv",
    mime="text/csv",
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Project: Career Progression and Promotion Gap Analysis "
    "for Retention Optimization"
)

st.caption(
    "Privacy note: EmployeeRecordID is a generated row reference, "
    "not a verified employee identifier. Review data permissions "
    "before sharing employee-level records."
)
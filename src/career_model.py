from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found inside the data folder."
    )

df = pd.read_csv(csv_files[0])

required_columns = [
    "Attrition",
    "Department",
    "JobRole",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobSatisfaction",
    "JobInvolvement",
    "WorkLifeBalance",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# This is a row reference, not a real employee ID.
df.insert(
    0,
    "EmployeeRecordID",
    np.arange(1, len(df) + 1),
)


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

company_years = df["YearsAtCompany"].replace(0, np.nan)

df["PromotionGapRatio"] = (
    df["YearsSinceLastPromotion"] / company_years
)

df["RoleStagnationIndex"] = (
    df["YearsInCurrentRole"] / company_years
)

df["TrainingIntensityScore"] = (
    df["TrainingTimesLastYear"] / company_years
)

df["ManagerTenureRatio"] = (
    df["YearsWithCurrManager"] / company_years
)


# --------------------------------------------------
# 4. PROMOTION GAP SCORE
# --------------------------------------------------

df["PromotionGapPoints"] = np.select(
    [
        df["YearsSinceLastPromotion"] >= 4,
        df["YearsSinceLastPromotion"] >= 2,
    ],
    [50, 30],
    default=10,
)

df["RoleStagnationPoints"] = np.select(
    [
        df["RoleStagnationIndex"].isna(),
        df["RoleStagnationIndex"] >= 0.75,
        df["RoleStagnationIndex"] >= 0.50,
    ],
    [15, 30, 20],
    default=5,
)

df["TrainingPoints"] = np.select(
    [
        df["TrainingTimesLastYear"] <= 1,
        df["TrainingTimesLastYear"] <= 2,
    ],
    [20, 10],
    default=5,
)

df["PromotionGapScore"] = (
    df["PromotionGapPoints"]
    + df["RoleStagnationPoints"]
    + df["TrainingPoints"]
).clip(0, 100)


df["PromotionGapCategory"] = np.select(
    [
        df["PromotionGapScore"] >= 70,
        df["PromotionGapScore"] >= 40,
    ],
    ["High", "Medium"],
    default="Low",
)


# --------------------------------------------------
# 5. RETENTION OPPORTUNITY INDEX
# --------------------------------------------------

# This is an intervention-priority heuristic,
# not a probability of resignation.

df["LowSatisfactionIndicator"] = np.where(
    df["JobSatisfaction"] <= 2,
    100,
    40,
)

df["LowTrainingIndicator"] = np.select(
    [
        df["TrainingTimesLastYear"] <= 1,
        df["TrainingTimesLastYear"] == 2,
    ],
    [
        100,
        50,
    ],
    default=20,
)

df["RetentionOpportunityIndex"] = (
    0.50 * df["PromotionGapScore"]
    + 0.25 * df["LowSatisfactionIndicator"]
    + 0.25 * df["LowTrainingIndicator"]
).round(2)


df["InterventionPriority"] = np.select(
    [
        df["RetentionOpportunityIndex"] >= 70,
        df["RetentionOpportunityIndex"] >= 45,
    ],
    ["High", "Medium"],
    default="Low",
)


# --------------------------------------------------
# 6. MACHINE LEARNING: K-MEANS CLUSTERING
# --------------------------------------------------

# Attrition is deliberately excluded from clustering.
# We are grouping career patterns, not predicting
# whether an employee will leave.

clustering_features = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobSatisfaction",
    "JobInvolvement",
    "WorkLifeBalance",
]

X = df[clustering_features].copy()

# StandardScaler requires finite values.
# The source dataset has no missing values, but
# ratio-derived features are not used for clustering.
if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError(
        "Clustering features contain missing or infinite values."
    )

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluate several cluster counts.
candidate_results = []
candidate_labels = {}

for k in range(2, min(7, len(df))):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    candidate_labels[k] = labels

    candidate_results.append({
        "k": k,
        "silhouette_score": score,
    })

if not candidate_results:
    raise ValueError(
        "Not enough records to evaluate clustering."
    )

evaluation_df = pd.DataFrame(candidate_results)

# Choose the candidate with the highest silhouette score.
best_k = int(
    evaluation_df.loc[
        evaluation_df["silhouette_score"].idxmax(),
        "k",
    ]
)

df["CareerCluster"] = candidate_labels[best_k]

evaluation_df.to_csv(
    OUTPUT_DIR / "clustering_evaluation.csv",
    index=False,
)

print("\n========== CLUSTER EVALUATION ==========")
print(evaluation_df.round(4).to_string(index=False))
print(f"\nSelected cluster count: {best_k}")


# --------------------------------------------------
# 7. CLUSTER SUMMARY
# --------------------------------------------------

cluster_summary = (
    df.groupby("CareerCluster", as_index=False)
    .agg(
        EmployeeCount=("EmployeeRecordID", "count"),
        AverageCompanyTenure=("YearsAtCompany", "mean"),
        AverageRoleTenure=("YearsInCurrentRole", "mean"),
        AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        AverageTraining=("TrainingTimesLastYear", "mean"),
        AverageJobSatisfaction=("JobSatisfaction", "mean"),
        HighGapEmployees=(
            "PromotionGapCategory",
            lambda s: (s == "High").sum(),
        ),
    )
)

cluster_summary = cluster_summary.round(2)

cluster_summary.to_csv(
    OUTPUT_DIR / "career_cluster_summary.csv",
    index=False,
)

print("\n========== CAREER CLUSTER SUMMARY ==========")
print(cluster_summary.to_string(index=False))


# --------------------------------------------------
# 8. SAVE EMPLOYEE-LEVEL RESULTS
# --------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "employee_career_scores.csv",
    index=False,
)

print("\n========== RISK CATEGORY DISTRIBUTION ==========")

print(
    df["PromotionGapCategory"]
    .value_counts()
    .to_string()
)

print("\n========== INTERVENTION PRIORITY ==========")

print(
    df["InterventionPriority"]
    .value_counts()
    .to_string()
)

print("\n========== OUTPUT FILES ==========")

print("1. employee_career_scores.csv")
print("2. career_cluster_summary.csv")
print("3. clustering_evaluation.csv")

print("\nCareer analysis and clustering completed.")
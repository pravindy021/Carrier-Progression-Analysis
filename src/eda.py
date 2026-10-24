from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


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
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


# --------------------------------------------------
# 3. DATA QUALITY CHECK
# --------------------------------------------------

print("\n========== DATA QUALITY ==========")

print(f"Dataset shape: {df.shape}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values: {df.isnull().sum().sum()}")

print("\nAttrition values:")
print(df["Attrition"].value_counts(dropna=False))

print("\nDepartment values:")
print(df["Department"].value_counts(dropna=False))


# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

# Avoid division by zero.
# Ratios are undefined when YearsAtCompany is zero.

df["PromotionGapRatio"] = (
    df["YearsSinceLastPromotion"]
    / df["YearsAtCompany"].replace(0, float("nan"))
)

df["RoleStagnationIndex"] = (
    df["YearsInCurrentRole"]
    / df["YearsAtCompany"].replace(0, float("nan"))
)

df["TrainingIntensityScore"] = (
    df["TrainingTimesLastYear"]
    / df["YearsAtCompany"].replace(0, float("nan"))
)

df["ManagerTenureRatio"] = (
    df["YearsWithCurrManager"]
    / df["YearsAtCompany"].replace(0, float("nan"))
)

# These are descriptive indicators, not validated risk predictions.


# --------------------------------------------------
# 5. CAREER METRICS SUMMARY
# --------------------------------------------------

career_columns = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "PromotionGapRatio",
    "RoleStagnationIndex",
    "TrainingIntensityScore",
    "ManagerTenureRatio",
]

print("\n========== CAREER METRICS SUMMARY ==========")

print(
    df[career_columns]
    .describe()
    .round(2)
)

# Save analysis-ready data.
# Original source CSV remains unchanged.

df.to_csv(
    OUTPUT_DIR / "employee_career_analysis.csv",
    index=False
)


# --------------------------------------------------
# 6. ATTRITION DISTRIBUTION
# --------------------------------------------------

attrition_counts = (
    df["Attrition"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

attrition_labels = {
    0: "Retained",
    1: "Left",
}

plt.figure(figsize=(8, 5))

sns.barplot(
    x=[attrition_labels[i] for i in attrition_counts.index],
    y=attrition_counts.values,
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Employee Status")
plt.ylabel("Number of Employees")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "attrition_distribution.png",
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 7. PROMOTION GAP DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="YearsSinceLastPromotion",
    bins=15,
)

plt.title("Years Since Last Promotion")
plt.xlabel("Years Since Last Promotion")
plt.ylabel("Number of Employees")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "promotion_gap_distribution.png",
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 8. DEPARTMENT-WISE PROMOTION GAP
# --------------------------------------------------

department_gap = (
    df.groupby("Department", as_index=False)
    .agg(
        EmployeeCount=("YearsSinceLastPromotion", "size"),
        AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        MedianPromotionGap=("YearsSinceLastPromotion", "median"),
        AverageRoleTenure=("YearsInCurrentRole", "mean"),
    )
    .sort_values("AveragePromotionGap", ascending=False)
)

print("\n========== DEPARTMENT CAREER ANALYSIS ==========")

print(department_gap.round(2).to_string(index=False))

department_gap.to_csv(
    OUTPUT_DIR / "department_career_summary.csv",
    index=False,
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=department_gap,
    x="Department",
    y="AveragePromotionGap",
)

plt.title("Average Years Since Last Promotion by Department")
plt.xlabel("Department")
plt.ylabel("Average Years Since Last Promotion")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "department_promotion_gap.png",
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 9. ROLE STAGNATION EXPLORATION
# --------------------------------------------------

role_summary = (
    df.groupby("JobRole", as_index=False)
    .agg(
        EmployeeCount=("YearsInCurrentRole", "size"),
        AverageRoleTenure=("YearsInCurrentRole", "mean"),
        AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        AverageTraining=("TrainingTimesLastYear", "mean"),
    )
    .sort_values("AverageRoleTenure", ascending=False)
)

print("\n========== JOB ROLE CAREER ANALYSIS ==========")

print(role_summary.round(2).to_string(index=False))

role_summary.to_csv(
    OUTPUT_DIR / "job_role_career_summary.csv",
    index=False,
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=role_summary,
    x="JobRole",
    y="AverageRoleTenure",
)

plt.title("Average Years in Current Role by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Average Years in Current Role")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "role_stagnation_analysis.png",
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 10. TRAINING AND CAREER PATTERNS
# --------------------------------------------------

training_summary = (
    df.groupby("TrainingTimesLastYear", as_index=False)
    .agg(
        EmployeeCount=("YearsSinceLastPromotion", "size"),
        AveragePromotionGap=("YearsSinceLastPromotion", "mean"),
        AverageRoleTenure=("YearsInCurrentRole", "mean"),
        AverageJobSatisfaction=("JobSatisfaction", "mean"),
    )
    .sort_values("TrainingTimesLastYear")
)

print("\n========== TRAINING PATTERN ANALYSIS ==========")

print(training_summary.round(2).to_string(index=False))

training_summary.to_csv(
    OUTPUT_DIR / "training_career_summary.csv",
    index=False,
)


# --------------------------------------------------
# 11. CORRELATION ANALYSIS
# --------------------------------------------------

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

print("\n========== CAREER FEATURE CORRELATIONS ==========")

career_correlation_columns = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobSatisfaction",
    "Attrition",
]

print(
    correlation.loc[
        career_correlation_columns,
        career_correlation_columns,
    ].round(2)
)

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation.loc[
        career_correlation_columns,
        career_correlation_columns,
    ],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)

plt.title("Career Metrics and Attrition Correlation")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "career_correlation_heatmap.png",
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 12. COMPLETION
# --------------------------------------------------

print("\n========== EDA COMPLETED ==========")

print(f"Charts and reports saved in: {OUTPUT_DIR}")

print("\nGenerated files:")

for file_path in sorted(OUTPUT_DIR.iterdir()):
    print(f"- {file_path.name}")
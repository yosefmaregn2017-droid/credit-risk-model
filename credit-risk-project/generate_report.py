from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# =========================
# PDF File path
# =========================
pdf_file = "Credit_Risk_Model_Report.pdf"

# =========================
# Report content
# =========================
report_text = """
Credit Risk Modeling Project – Final Report

Project Overview
This project aims to build a credit risk prediction model using transactional customer data. 
The objective is to identify high-risk customers and implement a robust pipeline from data processing to model deployment.

Task 1 – Data Exploration (EDA)
- Objective: Understand the structure, quality, and patterns in the data.
- Actions Taken:
  - Loaded raw transaction data.
  - Checked for missing values and data types.
  - Generated summary statistics and visualized distributions.
- Key Findings:
  - Missing values handled during preprocessing.
  - High variance in transaction amounts.
  - Some customers had very low engagement, highlighting potential high-risk cases.

Task 2 – Data Cleaning & Preprocessing
- Objective: Prepare raw data for feature engineering.
- Actions Taken:
  - Handled missing values (imputation) and removed duplicates.
  - Standardized date formats and encoded categorical variables.
- Outcome: Cleaned dataset stored in data/processed ready for feature engineering.

Task 3 – Feature Engineering
- Objective: Transform raw data into a model-ready format.
- Actions Taken:
  - Aggregate features: total, average, count, std of transactions.
  - Datetime features: hour, day, month, year extracted.
  - WOE & IV transformations for categorical features.
  - Built sklearn pipelines for automated preprocessing.
- Outcome: Processed features ready for modeling.

Task 4 – Proxy Target Variable Engineering
- Objective: Generate credit risk target variable (is_high_risk).
- Actions Taken:
  - Calculated RFM metrics.
  - Clustered customers with K-Means into 3 groups.
  - Identified high-risk cluster and created binary target.
- Outcome: Dataset now contains is_high_risk for supervised learning.

Task 5 – Model Training and Tracking
- Objective: Train models and track experiments using MLflow.
- Actions Taken:
  - Dropped identifier columns and one-hot encoded categorical features.
  - Split dataset into train/test sets.
  - Trained Logistic Regression, Random Forest, and Gradient Boosting models.
  - Evaluated models using accuracy, precision, recall, F1, and ROC-AUC.
  - Logged models and metrics to MLflow.
- Outcome: Complete MLflow experiment with trained models, ready for deployment.

Next Steps / Task 6
- Build FastAPI REST API for the model.
- Containerize using Docker.
- Implement CI/CD pipeline.
- Expose /predict endpoint for new customer data.

Project Folder Structure
credit-risk-project/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── features/
│   ├── modeling/
│   └── api/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore

Key Achievements
- Fully reproducible data processing and feature engineering pipelines.
- Proxy target variable successfully created.
- Models trained, evaluated, and tracked with MLflow.
- Project structured for deployment and CI/CD integration.
"""

# =========================
# Generate PDF
# =========================
doc = SimpleDocTemplate(pdf_file, pagesize=A4)
styles = getSampleStyleSheet()
story = []

for paragraph in report_text.split("\n\n"):
    story.append(Paragraph(paragraph.strip(), styles["Normal"]))
    story.append(Spacer(1, 12))

doc.build(story)
print(f"✅ PDF report generated: {pdf_file}")

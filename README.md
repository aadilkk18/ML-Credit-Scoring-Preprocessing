# ML-Credit-Scoring-Preprocessing



## Project Overview
This project implements an improved preprocessing pipeline for financial credit scoring, based on the 2024 research paper by Sayed et al. 

## Key Features
- **Iterative Imputation (MICE):** Advanced handling of missing values.
- **Robust Scaling:** Outlier-resistant feature scaling.
- **Yeo-Johnson Transformation:** Normalizing skewed financial data (Income, Credit Score).

## How to Run
1. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
2. Run the script: `python code/preprocessing_pipeline.py`

## Results
The pipeline successfully transforms skewed financial data into a Gaussian distribution, optimized for Deep Learning models.

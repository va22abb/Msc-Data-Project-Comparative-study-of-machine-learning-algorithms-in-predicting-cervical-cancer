# Msc-Data-Project-Comparative-study-of-machine-learning-algorithms-in-predicting-cervical-cancer

**Msc Data project thesis**

**Project Title**

Comparative-study-of-machine-learning-algorithms-in-predicting-cervical-cancer.

**Project Overview**

Cervical cancer prediction models play a crucial role in early detection and personalized care. This project compares machine learning models such as Random Forest, Support Vector Machine and XGBOOST to assess their effectiveness in predicting cervical cancer and also optimized with hyperparameter tuning . Using clinical and demographic data, the study will investigate how data preparation and feature selection impact model accuracy and reliabilit project also uses Explainable AI techniques, SHAP (SHapley Additive exPlanations), to highlight how individual features contribute to the model's predictions. This makes the model's decisions more transparent and helps in understanding the key factors influencing the outcomes. The ultimate goal is to improve screening and enable earlier diagnosis.

**Project Structure**

1.About Dataset: The dataset, sourced from the UCI Machine Learning Repository, contains 858 anonymised patient records from 2012–2013. It is in CSV format, GDPR-compliant, and ethically shared under a CC BY 4.0 license.

2.Dataset Overview and Loading: The dataset includes clinical, demographic, and medical data with "Biopsy" as the target. Data is loaded with Python’s pandas, addressing missing values and redundancies.

3.Data Preprocessing: Preprocessing involved handling missing values, converting categorical data to numerical forms, and addressing outliers. Irrelevant features were eliminated using Random Forest importance analysis.

4.Exploratory Data Analysis (EDA): EDA included descriptive statistics, correlation analysis, and visualisation using boxplots, histograms, and heatmaps. Records of individuals under 16 were excluded.

5.Model Selection and Training: Models used were SVM (with class balancing), Random Forest, and XGBoost (with imbalance handling). Features were scaled using StandardScaler.

6.Model Evaluation: Evaluation metrics included accuracy, F1-score, recall, confusion matrix, and ROC-AUC to assess performance.

7.Explainability and Interpretability Using SHAP Values: SHAP analysis clarified feature contributions to predictions, enhancing transparency for Random Forest and XGBoost outputs with summary plots.

**Data Overview**

This project uses a dataset containing 858 records on cervical cancer risk factors collected in Venezuela between 2012 and 2013. The dataset includes demographic, medical, and behavioral features such as age, number of sexual partners, contraceptive use, and history of STDs. The target variable is a Biopsy result indicating the presence or absence of cervical cancer. Ethically sourced from the UCI Machine Learning Repository, the data has been preprocessed for quality and accuracy, ensuring compliance with privacy standards and suitability for research.

**Research Question**

To what extent can machine learning algorithms be effectively optimized and compared for the accurate 
prediction of cervical cancer, and how do these models perform in terms of explainability, sensitivity, and 
clinical relevance?

**Dependencies**

Install or use any Python IDE preferably Google Colab or Jupyter notebook.

Install any required Python library packages for EDA, Model Development, and Performance Evaluation,preferably Confusion matrix(accuracy,precision,Recall and F1-Score) and AUC Score, ROC Curve.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn (including tools for SVM, Random Forest, StandardScaler, and evaluation metrics like confusion matrix and classification report).

**Result Short Note**

Three classification models—XGBoost, Random Forest Classifier, and Support Vector Classifier (SVC)—are evaluated based on their performance on a binary classification problem using unbalanced data. The best overall accuracy of 96% was shown by XGBoost, which performed exceptionally well in both Class 0 and Class 1 predictions, especially with a high recall and F1-score for the minority class. Strong results were also obtained using Random Forest, especially after hyperparameter adjustment, with an accuracy of 98% and enhanced Class 1 performance, demonstrating the best class balance. Although SVC did well for Class 0, its performance was poor for Class 1 forecasts, suggesting that its overall efficiency was limited. Metrics like confusion matrices, ROC curve analysis, and SHAP interpretability are included to show the advantages and disadvantages of each model. XGBoost and Random Forest are the most dependable options for dealing with class imbalance.

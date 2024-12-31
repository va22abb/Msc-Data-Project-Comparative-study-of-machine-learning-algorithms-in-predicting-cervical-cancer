# Msc-Data-Project-Comparative-study-of-machine-learning-algorithms-in-predicting-cervical-cancer

**Msc Data project thesis**

**Project Title**

Comparative-study-of-machine-learning-algorithms-in-predicting-cervical-cancer.

**Project Overview**

Cervical cancer prediction models play a crucial role in early detection and personalized care. This project compares machine learning models such as Random Forest, Support Vector Machine and XGBOOST to assess their effectiveness in predicting cervical cancer and also optimized with hyperparameter tuning . Using clinical and demographic data, the study will investigate how data preparation and feature selection impact model accuracy and reliabilit project also uses Explainable AI techniques, SHAP (SHapley Additive exPlanations), to highlight how individual features contribute to the model's predictions. This makes the model's decisions more transparent and helps in understanding the key factors influencing the outcomes. The ultimate goal is to improve screening and enable earlier diagnosis.

**Project Structure**

1.About Dataset: The dataset, sourced from the UCI Machine Learning Repository, contains 858 anonymised patient records from 2012–2013. It is in CSV format, GDPR-compliant, and ethically shared under a CC BY 4.0 license.

2.Dataset Overview and Loading: The dataset includes clinical, demographic, and medical data with "Biopsy" as the target. Data is loaded with Python’s pandas, addressing missing values and redundancies.

3.Handling Class Imbalance with SMOTE:To address class imbalance in the target variable, SMOTE was implemented to synthetically oversample the minority class. This improved the dataset's balance, ensuring better model training and evaluation.

4.Data Preprocessing: Preprocessing involved handling missing values, converting categorical data to numerical forms, and addressing outliers. Irrelevant features were eliminated using Random Forest importance analysis.

5.Exploratory Data Analysis (EDA): EDA included descriptive statistics, correlation analysis, and visualisation using boxplots, histograms, and heatmaps. Records of individuals under 16 were excluded.

6.Model Selection and Training: Models used were SVM (with class balancing), Random Forest, and XGBoost (with imbalance handling). Features were scaled using StandardScaler.

7.Model Evaluation: Evaluation metrics included accuracy, F1-score, recall, confusion matrix, and ROC-AUC to assess performance.

8.Explainability and Interpretability Using SHAP Values: SHAP analysis clarified feature contributions to predictions, enhancing transparency for Random Forest and XGBoost outputs with summary plots.

**Data Overview**

This project uses a dataset containing 858 records on cervical cancer risk factors collected in Venezuela between 2012 and 2013. The dataset has 36 features includes demographic, medical, and behavioral features such as age, number of sexual partners, contraceptive use, and history of STDs. The target variable is a Biopsy result indicating the presence or absence of cervical cancer. Ethically sourced from the UCI Machine Learning Repository, the data has been preprocessed for quality and accuracy, ensuring compliance with privacy standards and suitability for research.

**Research Question**

To what extent can machine learning algorithms be effectively optimized and compared for the accurate 
prediction of cervical cancer, and how do these models perform in terms of explainability, sensitivity, and 
clinical relevance?

**Dependencies**

Install or use any Python IDE preferably Google Colab or Jupyter notebook.

Install any required Python library packages for EDA, Model Development, and Performance Evaluation,preferably Confusion matrix(accuracy,precision,Recall and F1-Score) and AUC Score, ROC Curve.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn (including tools for SVM, Random Forest, StandardScaler, and evaluation metrics like confusion matrix and classification report).

**Result Short Note**

Three classification modelsX GBoost, Random Forest, and Support Vector Classifier (SVC) were tested to solve a binary classification problem with unbalanced data. Before applying SMOTE, XGBoost performed the best overall, achieving 96% accuracy and handling both classes well, especially the minority class with high recall and F1-scores. Random Forest followed closely with an impressive accuracy of 98%, showing strong performance across both classes after hyperparameter tuning. On the other hand, SVC worked well for the majority class (Class 0) but struggled to predict the minority class (Class 1), limiting its effectiveness.

After using SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset, all models showed improvements in handling the minority class. XGBoost and Random Forest both achieved 96% accuracy, maintaining strong and balanced performance across both classes. SVC also improved, achieving 94% accuracy and better predictions for the minority class. SMOTE helped reduce class imbalance, boosting recall and F1-scores for the minority class, particularly for SVC.

Overall, metrics like confusion matrices, ROC curves, and SHAP analysis highlighted the strengths and weaknesses of each model. While Random Forest and XGBoost remain the most reliable options, the use of SMOTE shows how addressing class imbalance can make all models fairer and more accurate.



# Msc-Data-Project-Comparative-study-of-machine-learning-algorithms-in-predicting-cervical-cancer

**Msc Data project thesis**

**Project Title**

Comparative-study-of-machine-learning-algorithms-in-predicting-cervical-cancer.

**Project Overview**

Cervical cancer prediction models play a crucial role in early detection and personalized care. This project compares machine learning models such as Random Forest, Support Vector Machine and XGBOOST to assess their effectiveness in predicting cervical cancer and also optimized with hyperparameter tuning . Using clinical and demographic data, the study will investigate how data preparation and feature selection impact model accuracy and reliabilit project also uses Explainable AI techniques, SHAP (SHapley Additive exPlanations), to highlight how individual features contribute to the model's predictions. This makes the model's decisions more transparent and helps in understanding the key factors influencing the outcomes. The ultimate goal is to improve screening and enable earlier diagnosis. SMOTE (Synthetic Minority Oversampling Technique Addictive) was also implemented to address class imbalance in datasets. This improved the dataset's balance and also ensured better model training and evaluation.

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

**Result**

Result and Analysis

This study evaluates the performance of three machine learning models—XGBoost, Random Forest Classifier, and Support Vector Classifier (SVC)—for a binary classification task with unbalanced data. The primary goal is to determine which model best handles the minority class while considering critical metrics such as accuracy, precision, recall, and F1-score. Additionally, confusion matrix analysis and ROC curve evaluation provide further insight into model discrimination capabilities. To improve prediction accuracy, Random Search hyperparameter tuning was employed, while SHAP analysis enhanced interpretability by identifying key feature contributions.

Classification Models Performance

The initial evaluation reveals that XGBoost outperforms the other models, achieving the highest overall accuracy (96%) and demonstrating balanced performance across both classes. SVC shows reliable results for Class 0 but struggles with Class 1, reflected in its lower F1-score for this class. Random Forest exhibits similar performance issues, especially in predicting the minority class.
Impact of SMOTE
Applying SMOTE (Synthetic Minority Over-sampling Technique) significantly enhances the recall for the minority class across all models. Random Forest and XGBoost show the most improvement, with better F1-scores and increased accuracy. Despite a minor precision loss, SMOTE effectively mitigates class imbalance, improving overall model robustness.

Hyperparameter Tuning and Final Comparison

After hyperparameter optimization, Random Forest achieves the best results with an overall accuracy of 98% and the highest F1-score for Class 1 (0.86). XGBoost follows closely, maintaining balanced performance. SVC, while improving slightly, remains less effective in handling the minority class. ROC curve analysis confirms that Random Forest and XGBoost have superior discriminative power, both achieving an AUC of 0.98.
Explainability with SHAP
SHAP analysis highlights key features influencing predictions, with Schiller and cytology being the most impactful. The SHAP summary and waterfall plots provide interpretability, aiding in understanding the models' decision-making processes. This transparency ensures reliable use in real-world applications such as cervical cancer risk prediction.


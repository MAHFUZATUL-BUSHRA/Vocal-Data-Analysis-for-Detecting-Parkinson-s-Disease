# Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease 
## Description:
Parkinson's disease affects the nervous system and lowers dopamine levels, which has an impact on the parts of the body that are controlled by nerves. By utilizing a technologically based tracking system, the Machine Learning (ML) approach can assist doctors in the decision-making process of diseases. Before seeing a doctor, patients can also complete screening tests, which strengthens the e-health system and costs less. Applying various speech analyzing processes to vocal data can help detect PD early because it affects the vocal signal very early in the progress of PD. In this study, I used voice recording datasets to apply seven machine learning classifiers. The classifiers are the XGBoost, Random Forest (RF), Linear Regression, Logistic Regression, Decision Tree, Support Vector Machine (SVM), and Neural Network (only on Dataset 2). The most accurate classifier overall is XGBoost, which outperformed all others with accuracy values of 1.00, 1.00, and 0.90 for Datasets 1, 2, and 3, respectively. For Datasets 1, 2, and 3, SVM has the second-highest accuracy score of 1.00, 0.98, and 0.89, in that order. PCA feature selection technique applied on Dataset 3 increased the accuracy rates of other classifiers for identifying early PD patients and highest for XGBoost Classifier. Due to improper disease evaluation and lack of treatment, it frequently results in the patient's death. The model is ideal for use in assisting with diagnosing Parkinson’s disease because of the model's greater accuracy of up to 100%.

## Objectives

I used three vocal datasets with varying numbers of features and voice recordings from the same individuals in each. The survey and contributions of this study included the following:
-To develop a machine learning model for predicting Parkinson’s disease using voice signals.


## Materials and Methods:
### A brief overview of the demonstrated method:
This section will provide an overview of the suggested framework, and details of every section in details. I have used google colab for the implementation. In the implementation process for diagnosis the Parkinson’s disease I have followed some steps as follows:
* Step 1: Import all the Python libraries that are required.
* Step 2: Load The voice datasets. Cleaning the missing data was the first step in the pre-processing, but it was only introduced as a precautionary measure because all datasets were already cleaned when they were collected. Since none of the rows contained any missing values, this step had little impact.
* Step 3: Fetch the features and targets from the DataFrame by dropping columns.
* Step 4: Scaling the inputs. (scale the feature data in the range of -1(minimum value) and 1(maximum value). Scaling is crucial because variables at various scales do not contribute equally to model fitting, potentially leading to bias. I fitted and transformed the feature data for preprocessing using "StandardScaler()" and "fit_transform ()".
* Step 5: Split the training and test data. Data are 80% used as training data and 20% as testing data.
* Step 6: Visualize Data for various knowledge and correlation between the features.
* Step 7: Apply different classifiers to the datasets. On the three datasets, six classifiers were used, and one more classifier was used on one of them.
* Step 8: Evaluate the performance. I use various performance matrices, such as precision, recall, f1-score, support, ROC Curve, and ROC AUC score, for evaluating and comparing the results.
* Step 9: Construct a predictive model that can find out whether or not they are affected by Parkinson's disease.

### Principal component analysis (PCA):
In dataset 3, the PCA feature selection algorithm is employed to improve the outcome and effectiveness of the classifications. And there are some notable changes. To explore information, and understand the variables and their relationships, PCA is used on multivariate data. This technique is useful for identifying outliers and condensing the dimensions of the feature space. PCA improves the performance of the model with higher Accuracy by producing independent, uncorrelated features of the dataset.  

### Classifications:
All datasets are subjected to the six supervised classifiers Linear Regression, Logistic Regression, Decision Tree, Support Vector Machine, Random Forest Classifier, and XGBoost Classifier, while dataset 2 is the sole exception applied Neural Network. Classifiers are built using the Sklearn libraries, neural networks are created using the Keras and Tensorflow libraries, other functions are developed using numpy and pandas, and various plots and graphical visualizations are generated using Seaborn, matplotlib, and Scikit-Plot.

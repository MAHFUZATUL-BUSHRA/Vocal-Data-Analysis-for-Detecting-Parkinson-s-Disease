# Vocal Data Analysis for Detecting Parkinson's Disease 
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

### Data Normalization and feature selection:
All three datasets are going through this pre-processing step. The process is a combination of data normalization and feature selection. The datasets have no null values. The feature selection method removes undesirable and unnecessary features from the dataset. In this method, the scaling method is used for generalized the data points by fitting the data points within a fixed scale as some of the applied classifiers are measured based on the distance of the data points. The Standardscaler method is used for feature scaling of the datasets. All three datasets have baseline features and dataset 3 has also other features named Mel frequency cepstral coefficients (MFCC), wavelet features (WT), and tunable Q-factor wavelet transform(TQWT). Feature selection technique principal component analysis (PCA) is used on dataset 3 for feature selection.
### Correlation Between the voice baseline features of Dataset 
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/correlation.png)
### Classifications:
All datasets are subjected to the six supervised classifiers Linear Regression, Logistic Regression, Decision Tree, Support Vector Machine, Random Forest Classifier, and XGBoost Classifier, while dataset 2 is the sole exception applied Neural Network. Classifiers are built using the Sklearn libraries, neural networks are created using the Keras and Tensorflow libraries, other functions are developed using numpy and pandas, and various plots and graphical visualizations are generated using Seaborn, matplotlib, and Scikit-Plot.

##  Confusion Matrix:
### Linear Regression:
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/linear.png)
### Logistic Regression:
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/logistic.png)
### Decision Tree:
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/Decision%20Tree.png)
### Support Vector Machine(SVM):
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/svm1.png)
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/svm2.png)
### Random Forest Classifier:
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/RF1.png)
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/RF2.png)
### XGBoost Classifier:
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/XG1.png)
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/XG2.png)
## ROC_AUC scores of the classifiers on different datasets
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/T1.png)
## ROC Curves of Different Classifiers:
### Dataset 1
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/t2.png)
### Dataset 2
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/r2.png)
### Dataset 3
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/R3.png)
### Dataset 3 With PCA
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/R4.png)
## Accuracy Metrics of Different Classifiers for Dataset 1
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/t3.png)
## Accuracy Metrics of Different Classifiers for Dataset 2
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/t4.png)
## Accuracy Metrics of Different Classifiers for Dataset 3
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/t5.png)
## Accuracy Metrics of Different Classifiers for Dataset 3 with PCA
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/t6.png)

## Performance Comparison of Multiple Classifiers On Different Vocal PD Datasets
![picture](https://github.com/MAHFUZATUL-BUSHRA/Vocal-Data-Analysis-for-Detecting-Parkinson-s-Disease/blob/main/pictures/classifiers.png)

The voice data is captured by smartphones which are used for detecting PD. This system can be useful for healthcare authorities if they can access the data of the older population (mostly common age above 60) and regularly monitor them and transfer the data to could-based signal processing. To predict PD, voice signal analysis employs a variety of classifiers, including XGBoost, Logistic Regression, Decision Tree, SVM, Random Forest, and Linear Regression. Since it is a neurological disease, neural networks are also utilized in this research. For each dataset, XGBoost performs as a perfect classifier with the highest accuracy, f1 score, precision, recall, and ROC_AUC score. Additionally, the majority of classifiers perform better when there are more recordings for each individual in the dataset because it is possible to classify individuals more accurately when there are more samples available. On the other hand, while datasets have more features, all classifiers perform relatively poorly. Patients with early Parkinson's disease participate in every dataset. All classifiers' performance (Linear Regression, Logistic Regression, Decision Tree, SVM, Random Forest) have significantly improved after applying the feature selection method PCA. By analyzing voice data, it is found that the Extreme Gradient Boost (XGBoost) technique should be applied to the PD prediction model. For Datasets 1 and 2, XGBoost accuracy is 100%, but for Dataset 3, accuracy is only 90%. For the XGBoost classifier, the PCA feature selection method is effective because it increases accuracy with 0.94.

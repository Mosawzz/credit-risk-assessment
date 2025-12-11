# Credit Risk Assessment: Classical Machine Learning vs Neural Network Models

## 📋 Project Overview

This project presents a comprehensive comparative analysis of multiple machine learning and deep learning models for credit risk classification. The goal is to predict whether a loan applicant is **High Risk** or **Low Risk** using demographic, financial, and loan-related data.

Both classical machine learning algorithms and neural network architectures are implemented, evaluated, and compared under a unified preprocessing and evaluation framework.

---

## 🎯 Objective

To determine the most effective model for credit risk prediction by:

* Applying standardized preprocessing steps.
* Training multiple classical ML and neural network models.
* Evaluating models using consistent performance metrics.
* Comparing accuracy, precision, recall, interpretability, and training efficiency.

---

## 📊 Dataset Information

* **Source:** Kaggle – Credit Risk Dataset
* **Initial Records:** 32,581
* **Features:** 11 independent variables
* **Target Variable:** `loan_status`

  * `0` → Low Risk
  * `1` → High Risk
* **Class Distribution:** Imbalanced (~22% High Risk)

---

## 🔑 Feature Groups

### **Personal Attributes**

* Age
* Income
* Employment Length
* Home Ownership

### **Loan Characteristics**

* Loan Amount
* Loan Grade
* Loan Intent
* Loan Percent of Income

### **Credit History**

* Previous Default History
* Credit History Length

---

## 🔧 Data Preprocessing Pipeline

### **3.1 Data Cleaning**

* Missing value imputation for `person_emp_length` using median.
* Removed records with excessive missing values in `loan_int_rate`.
* Removed duplicates.
* Outlier removal:

  * Age > 100
  * Income > $300,000
  * Employment Length > 50 years

### **3.2 Feature Engineering**

* **Binary Encoding**: `cb_person_default_on_file` (Y=1, N=0)
* **Ordinal Encoding**: `loan_grade` (A–G → 1–7)
* **One-Hot Encoding**: `person_home_ownership`, `loan_intent`

### **3.3 Data Splitting & Scaling**

* 80% Training, 20% Testing (Stratified)
* `StandardScaler` applied to 6 numerical features

---

## 🤖 Model Architectures

### **4.1 Classical Machine Learning Models**

* **Perceptron (PLA)** – default parameters
* **Logistic Regression** – max iterations: 1000
* **SVM (RBF Kernel)**:

  * Kernel: RBF
  * C = 1
  * gamma = scale
* **Random Forest Classifier**:

  * 200 estimators
  * Max depth: None
  * Random state: 42

### **4.2 Neural Network Models**

#### **Base Architecture**

* Input: 16 features
* Hidden layers: 32 → 16 neurons
* Activation: ReLU
* Output: 1 neuron (Sigmoid)
* Optimizer: Adam (lr = 0.001)
* Loss: Binary Crossentropy

#### **Final Optimized Architecture**

* Layers: 64 → 32 → Dropout(0.5) → 16 → 1
* Hidden Activation: ReLU
* Output Activation: Sigmoid
* Optimizer: Adam (0.001)
* Batch Size: 32
* Early Stopping (patience 5, restore best weights)

---

## 📈 Evaluation Framework

### **5.1 Evaluation Metrics**

* Accuracy
* Precision
* Recall
* F1-Score (Macro & Weighted)
* Confusion Matrix
* Training & Validation Curves
* PCA-based Decision Boundary

### **5.2 Validation Strategy**

* 20% test set
* 20% validation split for NN
* Early stopping enabled

---

## 📊 Results

### **6.1 Classical Machine Learning Performance**

| Model               | Accuracy | Precision | Recall | F1-Score | Key Observations            |
| ------------------- | -------- | --------- | ------ | -------- | --------------------------- |
| Perceptron          | 80.21%   | 0.54      | 0.63   | 0.58     | Lowest performance          |
| Logistic Regression | 86.04%   | 0.75      | 0.54   | 0.63     | Good precision, weak recall |
| SVM (RBF)           | 91.38%   | 0.93      | 0.66   | 0.77     | Excellent precision         |
| Random Forest       | 93.46%   | 0.96      | 0.73   | 0.83     | Best classical model        |

### **6.2 Neural Network Performance**

* Base NN Accuracy: **91.90%**
* Final Optimized NN Accuracy: **92.17%**
* Training Epochs: **19** (early stopping)

### **6.3 Model Comparison Summary**

| Aspect                | Best Classical (RF) | Neural Network    | Advantage  |
| --------------------- | ------------------- | ----------------- | ---------- |
| Accuracy              | 93.46%              | 92.17%            | Classical  |
| Precision (High Risk) | 96%                 | ~91%              | Classical  |
| Recall (High Risk)    | 73%                 | ~75%              | Neural     |
| Training Time         | Fast                | Moderate          | Classical  |
| Interpretability      | High                | Low               | Classical  |
| Decision Boundary     | Complex             | Highly Non-linear | Comparable |

---

## 🏆 Achievements & Enhancements

### **9.1 Model Enhancements**

#### Early Stopping

* Reduced epochs from 100 → ~20
* Prevented overfitting

#### Dropout Regularization (0.5)

* Improved generalization
* +1.5% validation accuracy

#### Automatic Best Weight Saving

* `restore_best_weights=True`

#### Performance Comparison

| Model Version | Test Accuracy | Validation Accuracy | Training Epochs |
| ------------- | ------------- | ------------------- | --------------- |
| Baseline      | 91.90%        | 91.63%              | 22              |
| Enhanced      | 92.17%        | 91.44%              | 19              |

### **9.2 GPU Training**

* Platform: Kaggle GPU (Tesla T4 / P100)
* Environment: Kaggle Notebook

### **9.3 Model Deployment**

* **Platform:** Streamlit Web Application
* **Features:**

  * 11-input form matching dataset
  * Automated preprocessing
  * Real-time prediction
  * Color-coded risk output
  * Confidence indicators

---

## 🧠 Final Conclusion

* Random Forest achieved the best overall classical performance.
* Neural Network showed stronger **recall** and good generalization.
* Classical models: faster, more interpretable.
* Neural networks: better nonlinear representation learning.
* Streamlit app enables real-time credit risk prediction with a clean, accessible interface.

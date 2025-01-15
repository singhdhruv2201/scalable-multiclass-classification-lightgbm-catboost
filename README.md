
## **Scalable Multi-Class Classification with LightGBM and CatBoostt**

### **Project Overview**
This project aims to solve a multi-class classification problem using a dataset of 1.2 million samples. The dataset was preprocessed, balanced, and used to train and evaluate two advanced machine learning models: **LightGBM** and **CatBoost**. The models were optimized and deployed using the **ONNX** runtime for efficient inference.

---

### **Table of Contents**
1. [Key Features](#key-features)
2. [Dataset Information](#dataset-information)
3. [Technologies Used](#technologies-used)
4. [Pipeline and Workflow](#pipeline-and-workflow)
5. [Model Evaluation](#model-evaluation)
6. [ONNX Conversion](#onnx-conversion)
7. [Results](#results)
8. [License](#license)
---

### **Key Features**
- Preprocessing of 1.2 million rows of raw data (scaling, cleaning, and feature selection).
- Addressed class imbalance using **Partial SMOTE** and class-weight adjustments.
- Trained **LightGBM** and **CatBoost** classifiers with hyperparameter tuning.
- Exported trained models in **ONNX format** for fast inference in production.
- Provided detailed evaluation with metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
- Designed clear visualizations of data distribution, feature importance, and model performance.

---

### **Dataset Information**
- **Number of Rows**: 1,200,000
- **Number of Features**: 15 (plus a target class column)
- **Target Classes**: 3 distinct categories.
- **Problem**: Multi-class classification with imbalanced classes.

---

### **Technologies Used**
- **Python**: Primary programming language.
- **Libraries**:
  - **Data Processing**: `pandas`, `numpy`, `scikit-learn`, `imblearn`
  - **Model Training**: `LightGBM`, `CatBoost`
  - **Deployment**: `ONNX`, `onnxmltools`, `onnxruntime`
  - **Visualization**: `Matplotlib`, `Seaborn`
- **Tools**:
  - Google Colab (with GPU for faster training)
  - ONNX Runtime for deployment.

---

### **Pipeline and Workflow**
1. **Data Preprocessing**:
   - Removed null values, scaled features, and mapped target labels.
2. **Train-Test Split**:
   - 80% data for training and 20% for testing.
3. **Class Balancing**:
   - Applied **Partial SMOTE** to oversample underrepresented classes.
4. **Feature Engineering**:
   - Analyzed feature importance using trained models.
5. **Model Training**:
   - Trained **LightGBM** and **CatBoost** with hyperparameter tuning.
6. **Model Evaluation**:
   - Evaluated models with metrics like accuracy, precision, recall, and confusion matrices.
7. **ONNX Conversion**:
   - Converted both models to **ONNX format** for lightweight, high-speed inference.

---

### **Model Evaluation**
#### LightGBM:
- **Accuracy**: ~72.2%
- **Precision (weighted)**: 74%
- **Recall (weighted)**: 72%
- **F1-Score (weighted)**: 71%

#### CatBoost:
- **Accuracy**: ~72.1%
- **Precision (weighted)**: 73%
- **Recall (weighted)**: 72%
- **F1-Score (weighted)**: 71%

---

### **ONNX Conversion**
Both the LightGBM and CatBoost models were successfully converted to **ONNX format** for deployment:
- **LightGBM ONNX Model**: `lightgbm_model_onnx.onnx`
- **CatBoost ONNX Model**: `catboost_model_onnx.onnx`

ONNX Runtime was used to verify predictions, ensuring consistent performance between the original models and their ONNX equivalents.

---

### **Results**
- Successfully built a pipeline for training multi-class classification models.
- Achieved over **72% accuracy** on both LightGBM and CatBoost models.
- Demonstrated the use of ONNX for efficient deployment and edge inference.


---

### **License**
This project is licensed under the **MIT License**.

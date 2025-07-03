# breast-cancer-svm-classification

A machine learning project using Support Vector Machines (SVMs) to classify breast cancer tumors as malignant or benign. This project demonstrates linear and non-linear SVMs, hyperparameter tuning, cross-validation, and decision boundary visualization using PCA.

---

### Objective

To build SVM-based models that accurately predict breast cancer diagnosis using clinical features. The models are evaluated on accuracy, cross-validation performance, and visual separation of classes in reduced 2D space.

---

### Dataset Description

The dataset contains features computed from digitized images of breast mass cell nuclei. Features include radius, texture, perimeter, area, smoothness, etc.

**Target Variable:**

* 1 → Malignant (cancerous)
* 0 → Benign (non-cancerous)

**Download Dataset**: [Click to view/download the dataset](breast-cancer.csv)

---

### Files Included

| File Name                     | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| `code(Task7).ipynb`           | Jupyter notebook containing all code, training, evaluation, and plots |
| `breast-cancer.csv`           | Dataset used in the project                                           |
| `Visualization using PCA.png` | Plot showing decision boundary using PCA                              |
| `README.md`                   | This project overview and instructions                                |

---

### Steps Covered in the Project

1. Data Preprocessing

   * Loaded dataset and removed the `id` column
   * Converted categorical `diagnosis` column to binary (M → 1, B → 0)
   * Standardized features using `StandardScaler`

2. SVM Classifiers

   * Trained both linear and RBF kernel SVMs
   * Evaluated model accuracy on test data
   * Compared performance between kernels

3. Hyperparameter Tuning

   * Used `GridSearchCV` to tune `C` and `gamma` for RBF kernel
   * Selected best model based on cross-validation score

4. Cross-Validation

   * Used `cross_val_score` with 5-fold CV to evaluate generalization

5. 2D Visualization with PCA

   * Reduced feature space to 2D using PCA
   * Trained SVM on PCA-reduced data (for visualization only)
   * Plotted decision boundary and data separation

---

### Key Takeaways

* RBF kernel can better model complex boundaries compared to linear SVM
* PCA is effective for visualizing high-dimensional separation
* Grid search helps find optimal SVM hyperparameters
* Cross-validation gives more reliable performance estimates

---

### How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the following files: `code(Task7).ipynb`, `breast-cancer.csv`
3. Run all cells sequentially to train models and generate outputs

---

### Visualizations

**View PCA Decision Boundary Plot**: [Visualization using PCA](Visualization%20using%20PCA.png)

---

### Future Enhancements

* Compare with Logistic Regression and Random Forest
* Add precision-recall and ROC curve evaluation
* Include support vector visualization
* Create a lightweight Streamlit app for predictions

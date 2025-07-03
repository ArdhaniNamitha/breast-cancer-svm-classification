# breast-cancer-svm-classification

This project uses Support Vector Machines (SVM) to classify breast tumors as malignant or benign using diagnostic features from digitized medical imaging. The implementation explores linear and non-linear classification, hyperparameter optimization, and dimensionality reduction for visualization.

---

### Objective

To develop and compare SVM models for tumor classification based on diagnostic data, with an emphasis on model interpretability and performance. The goal is to evaluate how kernel choice and parameter tuning affect classification, and to visualize model decisions in reduced feature space.

---

### Dataset Description

The dataset consists of diagnostic measurements collected from breast tissue samples. Each record includes 30 numerical features such as radius, perimeter, smoothness, and texture of cell nuclei derived from a digitized image.

**Target Variable:**

* `1` → Malignant (cancerous)
* `0` → Benign (non-cancerous)

**Download Dataset**: [Click to view/download the dataset](breast-cancer.csv)

---

### Files Included

| File Name                     | Description                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `code(Task7).ipynb`           | Main notebook containing all preprocessing, training, tuning, and visualization |
| `breast-cancer.csv`           | Dataset used in the classification models                                       |
| `Visualization using PCA.png` | 2D plot showing decision boundaries after PCA transformation                    |
| `README.md`                   | Detailed overview of the project, instructions, and observations                |

---

### Workflow Summary

1. **Data Preparation**

   * Removed unnecessary columns (`id`)
   * Converted diagnosis labels to binary format (M → 1, B → 0)
   * Standardized all features using `StandardScaler` to improve model performance

2. **Model Training (SVM)**

   * Trained two separate SVM models using:

     * Linear kernel
     * RBF (Gaussian) kernel
   * Compared the performance of each model on the test data

3. **Hyperparameter Optimization**

   * Used `GridSearchCV` to search over multiple values of `C` and `gamma` for the RBF model
   * Selected the model with the highest cross-validation score

4. **Cross-Validation**

   * Applied 5-fold cross-validation on the best-performing model
   * Reported mean accuracy across folds to validate model generalizability

5. **Visualization with PCA**

   * Performed dimensionality reduction using PCA (2 components)
   * Trained RBF SVM on PCA-reduced data
   * Plotted the decision boundary with class separation

---

### Key Insights

* Linear SVM provides fast and simple separation but may underperform on non-linear datasets
* RBF kernel adapts better to complex distributions and gives higher accuracy
* PCA helps in visualizing separability even if the model wasn't trained in 2D
* Hyperparameter tuning significantly improves model precision and generalization

---

### How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the following files: `code(Task7).ipynb`, `breast-cancer.csv`
3. Execute all code cells in the notebook to preprocess data, train models, and generate outputs

---

### Visualizations

**View PCA Decision Boundary Plot**: [Visualization using PCA](Visualization%20using%20PCA.png)

---

### Future Scope

* Add model comparison with Logistic Regression, Random Forest, and K-Nearest Neighbors
* Visualize support vectors and margin widths explicitly
* Integrate ROC curve and Precision-Recall metrics
* Build an interactive dashboard using Streamlit for live predictions

# 📊 Machine Learning Journey

> A practical exploration of Linear Regression techniques using the Advertising dataset

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

## 🎯 Project Overview

This repository contains hands-on implementations of linear regression models to predict sales based on advertising spend across different media channels (TV, Radio, and Newspaper). The project demonstrates both simple and multiple linear regression techniques, along with Stochastic Gradient Descent (SGD) optimization.

## 📁 Project Structure

```
.
├── advertising.csv          # Dataset containing advertising spend and sales data
├── Day2.ipynb              # Simple Linear Regression implementation
├── Day3.ipynb              # Multiple Linear Regression with SGD
└── README.md               # Project documentation
```

## 📚 Notebooks

### 📓 Day 2: Simple Linear Regression

**Topics Covered:**

- Data exploration and visualization
- Simple Linear Regression using TV advertising spend
- Train-test split methodology
- Model training and evaluation
- Performance metrics analysis

**Key Libraries:**

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` & `seaborn` - Data visualization
- `scikit-learn` - Machine learning models

**Highlights:**

- 📈 Pairplot visualization of feature relationships
- 🎯 Single feature (TV) prediction model
- 📊 80/20 train-test split
- 🔍 Model performance evaluation

---

### 📓 Day 3: Multiple Linear Regression with SGD

**Topics Covered:**

- Multiple feature linear regression
- Feature scaling with StandardScaler
- Stochastic Gradient Descent (SGD) Regressor
- Multiple variable prediction

**Key Concepts:**

- **Feature Scaling**: Standardization of features for optimal SGD performance
- **Multiple Features**: TV, Radio, and Newspaper advertising spend
- **SGD Optimization**: Iterative approach to find optimal parameters
- **Model Coefficients**: Understanding feature importance

**Model Configuration:**

```python
SGDRegressor(max_iter=5000, tol=1e-4, random_state=42)
```

## 📊 Dataset

**Advertising Dataset** contains:

- `TV`: Advertising spend on TV (in thousands of dollars)
- `Radio`: Advertising spend on Radio (in thousands of dollars)
- `Newspaper`: Advertising spend on Newspaper (in thousands of dollars)
- `Sales`: Product sales (in thousands of units)

**Dataset Size:** 200 observations

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repository-url>
   cd "Machine Learning"
   ```

2. **Install required packages**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

4. **Open and run the notebooks**
   - Start with `Day2.ipynb` for simple linear regression
   - Progress to `Day3.ipynb` for multiple linear regression

## 💡 Key Learnings

### Day 2 Insights

- ✅ Understanding the relationship between single variable and target
- ✅ Visualizing data patterns with pair plots
- ✅ Implementing train-test split for model validation
- ✅ Basic linear regression model building

### Day 3 Insights

- ✅ Importance of feature scaling in gradient descent
- ✅ Working with multiple features simultaneously
- ✅ SGD optimization for large datasets
- ✅ Interpreting model coefficients and intercept

## 🛠️ Technologies Used

| Technology                                                                                                        | Purpose                   |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------- |
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)                     | Programming Language      |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)                     | Data Manipulation         |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white)                        | Numerical Computing       |
| ![Scikit-learn](https://img.shields.io/badge/-Scikit%20Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine Learning          |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat&logo=python&logoColor=white)             | Visualization             |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat&logo=python&logoColor=white)                   | Statistical Visualization |

## 📈 Results

The models successfully predict sales based on advertising spend with:

- Simple Linear Regression: Single feature (TV) prediction
- Multiple Linear Regression: Multi-feature prediction with improved accuracy
- SGD implementation: Efficient optimization for parameter estimation

## 🔮 Future Enhancements

- [ ] Add polynomial regression
- [ ] Implement ridge and lasso regression
- [ ] Cross-validation techniques
- [ ] Feature engineering
- [ ] Model comparison dashboard
- [ ] Hyperparameter tuning
- [ ] Add more evaluation metrics (R², MSE, RMSE, MAE)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Kaif Zaki**

- GitHub: [@kaifzaki](https://github.com/kaifzaki)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show your support

Give a ⭐️ if this project helped you learn!

---

<div align="center">
  
**Happy Learning! 🚀**

Made with ❤️ and Python

</div>

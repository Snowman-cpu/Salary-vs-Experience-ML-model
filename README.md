# Supervised Simple Linear Regression

## Overview

This project implements **Simple Linear Regression** using Python. It aims to predict dependent variable values (such as salary) based on an independent variable (such as years of experience). The model is trained on a dataset and evaluated for performance.

## Dataset

The dataset used in this project is **Salary_Data.csv**, which contains:

- **Years of Experience** (Independent variable)
- **Salary** (Dependent variable)

## Prerequisites

Ensure you have the following libraries installed before running the code:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Installation & Usage

1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Supervised_Simple_Linear_Regression
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Supervised_Simple_Linear_Regression.ipynb
   ```

## Steps Involved

1. **Import Libraries**

   - `pandas` for data handling
   - `numpy` for numerical operations
   - `matplotlib` for visualization
   - `scikit-learn` for model building

2. **Load Dataset**

   ```python
   df = pd.read_csv('Salary_Data.csv')
   x = df.iloc[:, :-1].values  # Independent Variable
   y = df.iloc[:, -1].values   # Dependent Variable
   ```

3. **Splitting the Data**

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
   ```

4. **Train the Model**

   ```python
   from sklearn.linear_model import LinearRegression
   regressor = LinearRegression()
   regressor.fit(X_train, y_train)
   ```

5. **Predict Values**

   ```python
   y_pred = regressor.predict(X_test)
   ```

6. **Visualizing the Results**

   ```python
   plt.scatter(X_train, y_train, color='red')
   plt.plot(X_train, regressor.predict(X_train), color='blue')
   plt.title('Salary vs Experience (Training Set)')
   plt.xlabel('Years of Experience')
   plt.ylabel('Salary')
   plt.show()
   ```

## Model Evaluation

The model is evaluated using **R-Squared Score**:

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

## Results

- The regression line shows a linear relationship between **Years of Experience** and **Salary**.
- The **R-squared value** measures the accuracy of the model.

## References

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Pandas Documentation](https://pandas.pydata.org/)

## License

This project is licensed under the MIT License.

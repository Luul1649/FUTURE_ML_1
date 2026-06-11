# FUTURE_ML_1
# SALES AND DEMAND FORECASTING SYSTEM USING MACHINE LEARNING

###  Name

Haji Lul Ibrahim


---

# ABSTRACT

Sales forecasting is a critical business function that enables organizations to make informed decisions regarding inventory management, staffing, budgeting, and resource allocation. This project developed a Sales and Demand Forecasting System using Machine Learning techniques to predict future sales based on historical business data. The Sample Superstore dataset was used to train and evaluate different forecasting models including Random Forest, XGBoost, and Prophet.

The study involved data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and future sales forecasting. The performance of each model was assessed using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²). Results showed that Random Forest achieved the best predictive accuracy while Prophet provided valuable forecasting insights through trend and seasonality analysis. The developed system demonstrates how machine learning can support business decision-making by providing reliable sales forecasts.

Keywords: Sales Forecasting, Demand Forecasting, Machine Learning, Random Forest, Prophet, Time Series Analysis.

---

# CHAPTER ONE: INTRODUCTION

## 1.1 Background

Businesses operate in highly competitive environments where accurate forecasting plays a significant role in operational success. Sales forecasting helps organizations estimate future demand and make strategic decisions regarding inventory control, workforce planning, procurement, and financial management.

Traditional forecasting methods often rely on manual calculations and expert judgment, which may not adequately capture complex sales patterns. Machine Learning provides advanced techniques capable of learning from historical data and generating more accurate forecasts.

## 1.2 Problem Statement

Many businesses experience challenges such as:

* Overstocking products
* Stock shortages
* Poor inventory planning
* Revenue losses
* Inefficient staffing allocation

These challenges arise from inaccurate demand forecasting. Therefore, there is a need for an intelligent forecasting system capable of predicting future sales using historical business data.

## 1.3 Objectives

### General Objective

To develop a machine learning-based sales forecasting system that predicts future sales using historical business data.

### Specific Objectives

1. To collect and preprocess historical sales data.
2. To perform exploratory data analysis on sales trends.
3. To develop forecasting models using Machine Learning techniques.
4. To evaluate model performance using standard metrics.
5. To forecast future sales and provide actionable business insights.

## 1.4 Significance of the Study

The system assists businesses in:

* Improving inventory management.
* Reducing stock shortages.
* Optimizing cash flow.
* Enhancing operational efficiency.
* Supporting strategic decision-making.

---

# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Sales Forecasting

Sales forecasting refers to the process of estimating future sales based on historical data, market trends, and business intelligence.

## 2.2 Machine Learning in Forecasting

Machine Learning algorithms learn patterns from historical datasets and use these patterns to predict future outcomes. Common forecasting models include:

* Linear Regression
* Random Forest
* XGBoost
* ARIMA
* Prophet
* Long Short-Term Memory (LSTM)

## 2.3 Forecasting Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual values.

### Root Mean Squared Error (RMSE)

Measures the square root of average squared prediction errors.

### R-Squared (R²)

Measures the proportion of variance explained by the model.

---

# CHAPTER THREE: METHODOLOGY

## 3.1 Dataset

The Sample Superstore dataset was used in this study.

Dataset characteristics:

* Number of records: 9,994
* Number of features: 21
* Time period: 2014–2017

Important attributes included:

* Order Date
* Sales
* Quantity
* Discount
* Profit
* Region
* Category

## 3.2 Data Preprocessing

The following preprocessing activities were performed:

* Handling missing values
* Date conversion
* Data aggregation
* Feature extraction
* Creation of lag variables
* Creation of rolling averages

### Time Features Created

* Year
* Month
* Day
* Day of Week

### Lag Features

* Lag1
* Lag7
* Lag30

### Rolling Features

* Rolling7
* Rolling30
* Rolling Standard Deviation

## 3.3 Exploratory Data Analysis

Exploratory Data Analysis was performed to identify:

* Sales trends
* Seasonality
* Outliers
* Data distribution

The following visualizations were generated:

1. Daily Sales Trend
2. Monthly Sales Trend
3. Sales Distribution Histogram

## 3.4 Model Development

Three forecasting models were developed:

### Random Forest Regressor

Used as the baseline machine learning model.

### XGBoost Regressor

Used as an advanced ensemble learning model.

### Prophet

Used as a specialized time-series forecasting model capable of capturing trend and seasonality.

---

# CHAPTER FOUR: RESULTS AND DISCUSSION

## 4.1 Random Forest Results

Performance Metrics:

* MAE = 1726.53
* RMSE = 2441.08
* R² = 0.027

Interpretation:

The Random Forest model achieved the best predictive accuracy among all tested models. However, the low R² value indicates that daily sales exhibited substantial volatility.

## 4.2 XGBoost Results

Performance Metrics:

* MAE = 24420.17
* RMSE = 28999.56
* R² = -0.350

Interpretation:

The XGBoost model performed poorly due to the limited number of monthly observations available after aggregation.

## 4.3 Prophet Results

Performance Metrics:

* MAE = 1809.24
* RMSE = 2529.90

Interpretation:

Prophet achieved competitive forecasting performance while providing interpretable trend and seasonality components.

## 4.4 Comparative Analysis

| Model         | MAE      | RMSE     |
| ------------- | -------- | -------- |
| Random Forest | 1726.53  | 2441.08  |
| Prophet       | 1809.24  | 2529.90  |
| XGBoost       | 24420.17 | 28999.56 |

The Random Forest model achieved the lowest prediction error, making it the most accurate model in this study.

## 4.5 Forecasting Results

The Prophet model was used to forecast future sales for the next 30 days. Forecast outputs included:

* Predicted sales values
* Lower confidence intervals
* Upper confidence intervals

Trend and seasonality plots demonstrated recurring sales patterns over time.

---

# CHAPTER FIVE: CONCLUSION AND RECOMMENDATIONS

## 5.1 Conclusion

This project successfully developed a Sales and Demand Forecasting System using Machine Learning techniques. Historical sales data from the Sample Superstore dataset was analyzed and used to train forecasting models.

The Random Forest model achieved the highest predictive accuracy, while Prophet provided valuable insights into long-term sales trends and seasonality. The developed system demonstrates the practical application of Machine Learning in supporting business forecasting and decision-making.

## 5.2 Recommendations

Organizations should adopt forecasting systems to improve inventory planning and resource allocation.

Future enhancements may include:

* Real-time forecasting
* Deep Learning models such as LSTM
* Integration with ERP systems
* Sentiment analysis integration
* Multi-store demand forecasting

## 5.3 Future Work

Future research should investigate:

* Hybrid forecasting models
* Deep Learning approaches
* Real-time streaming data analytics
* Automated inventory optimization systems

---

# REFERENCES

1. Hastie, T., Tibshirani, R., & Friedman, J. (2021). The Elements of Statistical Learning.
2. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice.
4. Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. The American Statistician.
5. Prophet Documentation.
6. Scikit-Learn Documentation.
7. XGBoost Documentation.

---

# APPENDICES

Appendix A: Dataset Description

Appendix B: Python Source Code

Appendix C: Daily Sales Trend Graph

Appendix D: Monthly Sales Trend Graph

Appendix E: Prophet Forecast Plot

Appendix F: Prophet Trend and Seasonality Plot

Appendix G: Actual vs Predicted Sales Plot

# 🚂 Indian Railway Seat Demand Prediction

## 📌 Overview
This project uses **Decision Tree Classifier** to predict seat demand for Indian Railways. The model accurately classifies seat demand into three categories: **Low**, **Medium**, and **High** based on various travel parameters.

## 🎯 Problem Statement
Predicting seat demand helps railway authorities:
- Optimize seat allocation
- Implement dynamic pricing strategies  
- Improve passenger satisfaction
- Reduce empty seat wastage

## 📊 Model Performance
- **Algorithm:** Decision Tree (Entropy/Information Gain)
- **Accuracy:** 99.86%
- **Training Data:** 80,000+ samples
- **Test Data:** 20,000 samples

## 🔍 Features Used
| Feature | Description | Range |
|---------|-------------|-------|
| Train Type | Type of train | Express, Passenger, Rajdhani, Shatabdi, Superfast |
| Seat Type | Class of seat | General, Sleeper, AC1, AC2, AC3 |
| Season | Travel season | Off, Normal, Peak |
| Distance | Journey distance (km) | 50-2500 km |
| Is Holiday | Holiday indicator | Yes/No |
| Is Weekend | Weekend indicator | Yes/No |
| Base Price | Ticket price (₹) | 100-5000 |

## 📈 Demand Categories
- **🔴 High Demand** - Seats fill quickly, book immediately
- **🟢 Low Demand** - Good availability, flexible booking
- **🟡 Medium Demand** - Moderate demand, book 1-2 weeks ahead

## 🛠️ Technologies Used
- Python 3.12
- Scikit-learn (Decision Tree Classifier)
- Streamlit (Web Interface)
- Pandas, NumPy
- Joblib (Model Persistence)
- Matplotlib, Seaborn (Visualizations)

## 📁 Project Structure

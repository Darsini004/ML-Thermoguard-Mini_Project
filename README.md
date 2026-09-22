# ML-Thermoguard-Mini_Project
# 🔥 THERMOGUARD – Machine Learning Based Thermal Throttling Prediction System

**Domain:** Data Science using Python  
**Algorithm:** Gaussian Naive Bayes  
**Presented by:** Darsini Rajendran

---

## 📌 Project Overview

**ThermoGuard** is a Machine Learning based system that predicts whether a computer system may experience **thermal throttling**.

The system analyzes parameters such as CPU usage, RAM usage, GPU usage, CPU/GPU temperature, ambient temperature, and fan speed. A **Gaussian Naive Bayes** model learns patterns from historical data and predicts the thermal throttling status as **Yes / No**.

---

## 🎯 Objectives

- Predict possible thermal throttling using Machine Learning.
- Analyze CPU, RAM, GPU and temperature-related parameters.
- Reduce manual analysis of system conditions.
- Provide a quick **Yes / No** prediction.
- Demonstrate Gaussian Naive Bayes for a real-world classification problem.

---

## 🧠 Machine Learning Algorithm

### Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm used when the input features are continuous and approximately follow a Gaussian/normal distribution.

### Gaussian Probability Formula

```text
P(x|C) = 1 / √(2πσ²) × e^(-(x−μ)² / 2σ²)

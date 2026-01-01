# Gaussian Naive Bayes (Python)

This repository contains a ** implementation of the Gaussian Naive Bayes (GNB) classification algorithm** using **Python and NumPy**

The project demonstrates how continuous features can be modeled using **Gaussian (normal) distributions** and applied to a simple real-world classification problem.

---

## 📌 Problem Description

Given a small dataset containing physical attributes of people:
- Height (cm)
- Weight (kg)
- Foot size (cm)

Each person is labeled as either **Male** or **Female**.

The goal is to:
- Train a Gaussian Naive Bayes classifier using this dataset
- Allow the user to input a new sample (height, weight, foot size)
- Predict whether the person is **Male** or **Female**

---

## 🧠 Theory Overview

Gaussian Naive Bayes is a **probabilistic classification algorithm** based on **Bayes’ Theorem**, with the assumptions that:
1. Features are **conditionally independent** given the class
2. Continuous features follow a **Gaussian (normal) distribution**

### Bayes’ Theorem 

Posterior Probability ∝ Prior Probability × Likelihood

P(y | X) = P(y) × Π P(xᵢ | y)

Where:
- P(y) is the prior probability of class y
- P(xᵢ | y) is the likelihood of feature xᵢ given class y
- Π means multiplication over all features
- The class with the highest posterior probability is selected as the prediction

---

## ⚙️ Implementation Details

- **Language**: Python  
- **Libraries used**: NumPy (for mathematical operations)
- **No high-level ML libraries** are used
- The algorithm is **non-iterative** (no training loops or stopping criteria)

### Main Steps:
1. Separate data by class
2. Compute mean and variance for each feature per class
3. Compute Gaussian likelihoods
4. Compute posterior probabilities
5. Predict the class with the highest posterior

---

## 🧪 Dataset Used

| Gender | Height (cm) | Weight (kg) | Foot Size (cm) |
|------|------------|-------------|---------------|
| Male | 180 | 80 | 27 |
| Male | 175 | 78 | 26 |
| Male | 170 | 72 | 25 |
| Male | 178 | 75 | 27 |
| Female | 160 | 55 | 23 |
| Female | 165 | 60 | 24 |
| Female | 155 | 50 | 22 |
| Female | 162 | 58 | 23 |

---
## 👥 Group Members

| No. | Name              | ID           |
|----:|-------------------|--------------|
| 1   | Abel Getachew     | UGR/6211/15  |
| 2   | Dagmawi Heywot    | UGR/4392/15  |
| 3   | Geleta Tamiru     | UGR/2035/15  |
| 4   | Nathnael Lule     | UGR/1003/15  |

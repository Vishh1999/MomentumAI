# MomentumAI

MomentumAI is an end-to-end ML + AI project that demonstrates how workout data can be turned into actionable, post workout coaching insights.  
It combines traditional machine learning models with a lightweight LLM layer to generate user facing feedback for a hybrid fitness application.

---

## Project Overview

The system works in three stages:

1. **Model Training**
   - Workout history is used to train classification models.
   - Two prediction tasks are learned:
     - Whether the *next workout* is likely to include a personal best.
     - Whether the *training streak* is at risk of breaking.

2. **Model Inference**
   - The best performing model is saved and reused.
   - A user’s latest workout data is passed through the trained models to generate probabilities.

3. **LLM Based Coaching Output**
   - Model outputs are converted into simple product signals.
   - An LLM generates short, post-workout coaching messages shown immediately after a session.

The LLM is used only for communication, not for prediction.

---
## Models Used

Three classifiers were evaluated:

- Logistic Regression (baseline)
- Random Forest (Medium)
- XGBoost (Advanced)

Logistic Regression was selected for deployment because it performed slightly better on accuracy while remaining simple, interpretable, and stable.

---

## Features Used

The models use a small set of interpretable features:

- Training load (distance * average speed)
- Average heart rate
- Moving time
- Current workout streak length

---

## Dataset

The dataset used in this project is a public Strava activity dataset sourced from Kaggle.

- Data includes distance, duration, speed, heart rate, and personal record counts.
- The data was lightly processed for feature engineering and modeling.
- All credit for the dataset belongs to the original Kaggle contributor.

This project is for learning and demonstration purposes only.

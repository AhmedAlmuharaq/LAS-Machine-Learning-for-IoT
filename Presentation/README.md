# Course Presentation — LAS Machine Learning for IoT

**File:** `LAS_Machine_Learning_for_IoT.pptx`  
**Author:** Michel Salomon  
**Institution:** Université Marie & Louis Pasteur (UMLP)  
**Program:** Master 1 — LAS (Embedded Computing Systems / IoT)  
**Date:** May–June 2026  

---

## Overview

This PowerPoint presentation defines the course structure, lab assignments, and task descriptions for the **Machine Learning for IoT** module within the Master 1 LAS program.

---

## Course Schedule

| Date | Supervisor | Topic |
|------|-----------|-------|
| Tuesday, May 26, 2026 | Michel Salomon | Software labs (Lab 1 & Lab 2) + Task descriptions |
| Thursday, May 28, 2026 | Michel Salomon | Task 1 (SNCF IoT data) + Task 2 (Smart City) |
| Wednesday, May 27, 2026 | Jacques Bahi | Stock exchange prediction and time series |

---

## Labs Description

### Lab 1 — Logistic Regression and Metrics
A warm-up lab to practice:
- Loading and exploring a dataset
- Splitting data into train/test with stratification
- Training a Logistic Regression classifier
- Evaluating with precision, recall, F1-score, and confusion matrix

See: [Lab 1 Folder](../Lab1_Logistic_Regression/)

### Lab 2 — Decision Trees and Random Forests
A regression lab covering:
- Decision Tree regressor with depth tuning
- Random Forest regressor and feature importance
- Ensemble stacking with meta-models

See: [Lab 2 Folder](../Lab2_Decision_Trees_Random_Forests/)

---

## Tasks Description

### Task 1 — IoT Data from LAS Realisation
Apply machine learning to data collected during the previous LAS semester's IoT project supervised by Joseph Azar. Students could also propose new ML approaches on their collected IoT data.

See: [Task 1 Folder](../Task1_SNCF_Punctuality/)

### Task 2 — Smart City: Urban Noise Monitoring
Reproduce results from the paper:
> *"Deep Learning and Gradient Boosting for Urban Environmental Noise Monitoring in Smart Cities"*

**Required pipeline (12 steps):**
1. Basic imports
2. Load data from 3 parkmeters
3. Compute LAeq from 77 histogram bins (Equation 1)
4. Resample to 1-hour intervals (May 2019 – Jan 2020)
5. Visualize hourly noise trends (Figure 2)
6. Normalize with MinMaxScaler + train/test split
7. Create supervised dataset with look_back parameter
8. Train LightGBM — Config 1 (Table 2, upper line)
9. Train LightGBM — Config 2 (Table 2, lower line)
10. Calculate RMSE (inverse-scaled)
11. 6-day forecasting (Figure 4)
12. Compare with alternative ML models

See: [Task 2 Folder](../Task2_SmartCity_Noise/)

---

## Submission

**Deadline:** Saturday, June 6, 2026 — 6 PM  
**Contact:** michel.salomon@umlp.fr  
All Colab notebooks (Labs + Tasks) must be submitted by email.

# 🕵SMOTE Deep Analysis — Oversampling Techniques for Imbalanced Fraud Data (No Modeling)

This repo is a **hands-on, visual deep dive** into **oversampling** for imbalanced datasets using the Credit Card Fraud data.  
It **does not train any models**. Instead, it focuses on **how different oversampling methods change the data itself** and what those changes imply for downstream modeling.

> **Pipeline covered:** Baseline (no balancing) ➜ **Random Oversampling (ROS)** ➜ **SMOTE** ➜ **Borderline-SMOTE (1 & 2)** ➜ **KMeans-SMOTE**  
> You’ll see class counts before/after, distribution shifts, and practical guidance on when to use which method.

---

## Why This Exists

In highly imbalanced problems (like fraud detection), most tutorials jump straight to “apply SMOTE and train a model.”  
This notebook goes one layer deeper: **what exactly happens to the data when we oversample**, and how do different methods **reshape decision boundaries**?

If you understand the **data geometry** these methods create, you’ll make better choices later when you *do* train models.

---

## Dataset

I used the **Credit Card Fraud Detection Dataset** 

- File used in notebook: `creditcard.csv`  
- Size: ~150 MB (too large to upload here). 

Download directly from Kaggle: [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

---

## Techniques Compared (Concept & Effects)

### 1) No Oversampling (Baseline)
- Imbalance remains extreme (e.g., `0: 284,315 | 1: 492`).
- Any classifier trained later tends to predict “majority” most of the time.
- **Use as a reference** to see what changes each method introduces.

### 2) Random Oversampling (ROS)
- **Duplicates** existing minority samples until balanced.
- Simple, fast, preserves the original minority manifold.
- **Overfitting risk** (no new information; same points repeated).
- Good as a **baseline** to compare against synthetic methods.

### 3) SMOTE (Regular)
- **Interpolates** between minority neighbors to generate **new** samples.
- Adds variation (helps generalization more than ROS).
- Can create **unrealistic samples** in sparse regions.
- Solid default when you first try synthetic oversampling.

### 4) Borderline-SMOTE
- Focuses only on **“danger zone”** minority points near the class boundary.
- **Borderline-1:** Generates minority samples near the boundary.
- **Borderline-2:** Generates even “harder” samples closer to majority.
- Emphasizes **difficult regions**, potentially boosting recall on tricky cases.
- More sensitive to noise near the boundary.

### 5) KMeans-SMOTE
- Applies **clustering** first, then oversamples **more** in minority-heavy clusters and **less** in majority-heavy ones.
- **Adaptive**: avoids oversampling “easy” regions; encourages **diverse** synthetic samples.
- Slightly more setup (cluster count, density exponent).

---

## What You’ll See In the Notebook

- **Class counts** before/after each method (e.g., ROS balancing to 1:1).
- **Distribution/geometry changes** via 2D projections/plots (e.g., PCA/t-SNE-style visuals if used).
- **Side-by-side comparisons** of SMOTE variants with short, practical notes.
- **Final summary** explaining when to pick ROS vs SMOTE vs Borderline vs KMeans-SMOTE.

> The goal is to build **intuition** for how each technique reshapes the dataset — not to benchmark models.



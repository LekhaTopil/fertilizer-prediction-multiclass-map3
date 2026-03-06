# **Optimal Fertilizer Prediction: Multi-Class Classification with MAP@3**
![alt text](https://img.shields.io/badge/Python-3.11.13-blue)   ![alt text](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM-green)

# 📌 **Project Overview**
This project addresses a multi-class classification challenge to predict the most suitable fertilizer for various agricultural scenarios. Using a dataset of 1 million records (750k train / 250k test), the objective is to rank the **top 3 fertilizer recommendations** based on environmental and soil-related features.
The solution focuses heavily on advanced feature engineering and gradient boosting frameworks to capture complex interactions between soil nutrients, crop types, and climate conditions.

# 📊 **Dataset Description**
- **Total Records:** 1,000,000 (750,000 Training | 250,000 Test)
- **Features:** 8 base features (excluding id and fertilizer_name).
- **Target Variable:** **`fertilizer_name`** (**7** distinct categories).

# 📈 **Evaluation Metric**
The performance is measured using **Mean Average Precision at 3 (MAP@3).** This metric emphasizes:
- Identifying the correct fertilizer.
- The quality of the ranking for the top 3 recommendations.

# 🛠 **Feature Engineering (The Core Approach)**
Since the dataset is synthetic, significant effort was placed on introducing real-world agronomic complexity through engineered features:
## **1. Contextual Ratios (Social Balance)**
I created ratios to measure the social balance of an individual's lifestyle.
- **social_vs_alone_ratio:** Comparing social attendance to time spent alone helps identify if an individual seeks external stimulation or internal reflection, regardless of how busy their overall schedule is.
- **alone_to_friends_ratio:** This measures the depth vs. breadth of social connections. A large friend circle with high time spent alone suggests a different social profile than a small circle with low alone time.
- **posting_freq_vs_social_event_ratio:** This distinguishes between "Digital Extroversion" (social media) and "Physical Extroversion" (event attendance).
  
## **2. Domain-Specific Interactions**
- **Composite Features:** Created **`soil_crop_combo`** by combining soil and crop types. This became the most influential feature in the model.
- **Nutrient Ratios:** Engineered **NPK-derived** features including **`npk_balance`**, **`n_p_ratio`**, **`n_k_ratio`**, and **`p_k_ratio`**.
- **Climate Stress Indicators:** Features like **`hot_dry_conditions`** and **`ideal_growing_conditions`** were derived from temperature and moisture data.

## **3. Probability Encoding**
- **Group-wise Likelihoods:** Created a function to calculate the mean distribution of fertilizer labels within specific soil_crop_combo groups.
- **Leakage Prevention:** These probabilities were computed strictly within a **5-Fold Stratified K-Fold cross-validation loop** to ensure the model generalizes to unseen data and to capture historical tendencies without data leakage.


# 🤖 **Modeling & Validation Strategy**
## **Validation:** 
 - Used a 5-Fold Stratified K-Fold strategy (maintaining class balance across folds).

## **Algorithms**
- **LightGBM** and **XGBoost** were used for model training and evaluation.
- Both models performed competitively, but **LightGBM showed slightly better cross-validation stability and Private Leaderboard performance**, and was therefore selected for final predictions.


# 🔍 **Key Insights & Feature Importance**
- The combined feature **soil_crop_combo** (soil type x crop type) emerged as the most influential factor in fertilizer recommendation.
- Engineered **NPK ratio** and **interaction features** (e.g., **npk_balance, nutrient ratios,** and **climate-nutrient** interactions) significantly improved model performance.
- Fertilizer probability target encoding based on **soil-crop** groups showed limited benefit, likely due to the synthetic dataset structure.

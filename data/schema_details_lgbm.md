## 1. Feature Engineering

These functions are our toolkit. They are designed to extract temporal dynamics, handle missing demographics (the "Cold Start" user), and encode high-cardinality IDs safely.

Below is a comprehensive summary of all the features engineered in this pipeline, categorized by their underlying methodology.

**Feature Glossary & Data Dictionary**

| Feature Name | Category | Source | Description / Engineering Method |
| :--- | :--- | :--- | :--- |
| **`pvalue_level`** | Demographics | Original (Imputed) | Purchasing power level. Imputed using Category-Based Mode (most frequent value for the current `cate_id`). Fallbacks to `-1`. |
| **`age_level`** | Demographics | Original (Imputed) | User age bracket. Imputed using Category-Based Mode. Fallbacks to `-1` for true cold-start. |
| **`new_user_class_level`** | Demographics | Original (Imputed) | City tier or new user indicator. Imputed using Category-Based Mode. Fallbacks to `-1`. |
| **`cms_group_id`** | Demographics | Original (Imputed) | User demographic profile group. Imputed using Category-Based Mode. Fallbacks to `-1`. |
| **`occupation`** | Demographics | Original (Imputed) | User occupation status. Imputed using Category-Based Mode. Fallbacks to `-1`. |
| **`shopping_level`** | Demographics | Original (Imputed) | E-commerce participation depth. Imputed using Category-Based Mode. Fallbacks to `-1`. |
| **`cms_segid`** | Demographics | Original (Imputed) | Micro-segmentation profile ID. Imputed using Category-Based Mode. Fallbacks to `-1`. |
| **`final_gender_code`** | Demographics | Original (Imputed) | User gender. Imputed using Category-Based Mode. Fallbacks to `-1`. |
| **`price`** | Financial | Original (Processed)| The cost of the ad item, transformed using `log1p` to normalize high-variance financial outliers. |
| **`time_since_last_click`** | Temporal | Engineered | Seconds elapsed since the user's last logged event. Measures ad fatigue and intent decay. |
| **`hour`** | Temporal | Engineered | The hour of the day (0-23) extracted from localized `event_time` to capture diurnal shopping cycles. |
| **`weekday`** | Temporal | Engineered | The day of the week (0-6) extracted from localized `event_time` to capture weekly shopping cycles. |
| **`user_count_1H`** | Momentum | Engineered | Rolling total of past interactions over a strictly past 1-hour window. Left-closed to prevent temporal leakage. |
| **`user_count_1D`** | Momentum | Engineered | Rolling total of past interactions over a strictly past 1-day window. Left-closed to prevent temporal leakage. |
| **`user_count_3D`** | Momentum | Engineered | Rolling total of past interactions over a strictly past 3-day window. Left-closed to prevent temporal leakage. |
| **`user_ctr_1H`** | Momentum | Engineered | Rolling average CTR over a strictly past 1-hour window. Captures immediate "clicking sprees." |
| **`user_ctr_1D`** | Momentum | Engineered | Rolling average CTR over a strictly past 1-day window. |
| **`user_ctr_3D`** | Momentum | Engineered | Rolling average CTR over a strictly past 3-day window. |
| **`brand_freq`** | ID Encoding | Engineered | Frequency Encoding: Replaces raw hashed Brand ID with its overall dataset occurrence count (Popularity). |
| **`cate_id_freq`** | ID Encoding | Engineered | Frequency Encoding: Replaces raw hashed Category ID with its overall dataset occurrence count. |
| **`cms_group_id_freq`** | ID Encoding | Engineered | Frequency Encoding: Replaces raw hashed Group ID with its overall dataset occurrence count. |
| **`pid_freq`** | ID Encoding | Engineered | Frequency Encoding: Replaces raw hashed Placement ID with its overall dataset occurrence count. |
| **`brand_te`** | ID Encoding | Engineered | Expanding Window Target Encoding for Brand ID. Cumulative historical CTR calculated using strictly preceding time-steps. |
| **`cate_id_te`** | ID Encoding | Engineered | Expanding Window Target Encoding for Category ID. Cumulative historical CTR to prevent future-to-past leakage. |
| **`cms_group_id_te`**| ID Encoding | Engineered | Expanding Window Target Encoding for Group ID. Cumulative historical CTR. |
| **`pid_te`** | ID Encoding | Engineered | Expanding Window Target Encoding for Placement ID. Cumulative historical CTR. |
| **`brand_be`** | ID Encoding | Engineered | Bayesian Smoothing for Brand ID. CTR estimate shrunk toward the global mean to stabilize rare brand noise. |
| **`cate_id_be`** | ID Encoding | Engineered | Bayesian Smoothing for Category ID. Stabilizes categories with misleading 100% or 0% raw CTRs. |
| **`cms_group_id_be`**| ID Encoding | Engineered | Bayesian Smoothing for Group ID. |
| **`pid_be`** | ID Encoding | Engineered | Bayesian Smoothing for Placement ID. |
| **`ad_ctr_cv`** | High-Cardinality CTR | Engineered | 5-Fold OOF average CTR specifically generated for the ultra-high cardinality `adgroup_id`. |
| **`user_ctr_cv`** | High-Cardinality CTR | Engineered | 5-Fold OOF average CTR specifically generated for the ultra-high cardinality `user_id`. |
| **`cate_id_filtered`** | Interaction Base | Engineered | Retains only the Top 200 most frequent categories; groups all rare ones as `-1` to reduce cross-feature noise. |
| **`gender_cate`** | Categorical Interactions| Engineered | String-concatenated cross feature (`final_gender_code` x `cate_id_filtered`). Capped at Top 150 combinations. |
| **`age_cate`** | Categorical Interactions| Engineered | String-concatenated cross feature (`age_level` x `cate_id_filtered`). Capped at Top 150 combinations. |
| **`brand_cate`** | Categorical Interactions| Engineered | String-concatenated cross feature (`brand` x `cate_id_filtered`). Capped at Top 150 combinations. |
| **`affinity_score`** | Category Affinity | Engineered | Weighted sum of historical category interactions (`buy`=5, `cart`=3, `fav`=2, `pv`=1). Uses temporal-shield in BigQuery. |
| **`total_interactions`** | Category Affinity | Engineered | Total raw count of past historical interactions within the category from the behavior log. |
| **`interest_recency`** | Category Affinity | Engineered | Time delta between the current ad impression and the most recent historical interaction in the behavior log (Intent Decay). |
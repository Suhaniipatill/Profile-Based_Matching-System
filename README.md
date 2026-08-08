# 🤝 Profile Based Matching and Recommendation System

A Machine Learning based profile matching and recommendation system that recommends the **Top 5 most compatible user profiles** using a hybrid similarity approach.

## 📌 Project Overview

The system analyzes professional user profiles and calculates compatibility using:

- **TF-IDF Text Similarity**
- **Cosine Similarity**
- **MBTI Personality Matching**
- **Location Matching**

The final compatibility score combines these factors to rank and recommend the most suitable profiles.

## 🎯 Objectives

- Develop a profile-based recommendation system using Machine Learning.
- Compare professional profiles using text similarity.
- Incorporate personality and location as additional compatibility factors.
- Provide an interactive web interface for profile selection and recommendations.
- Collect Accept/Reject feedback for future improvements.

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **TF-IDF**
- **Cosine Similarity**

## ⚙️ Methodology

The recommendation process follows these steps:

```text
User Dataset
     ↓
Data Preprocessing
     ↓
Profile Text Creation
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity
     ↓
MBTI Matching
     ↓
Location Matching
     ↓
Hybrid Compatibility Score
     ↓
Ranking
     ↓
Top 5 Recommendations

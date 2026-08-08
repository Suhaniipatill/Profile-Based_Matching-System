import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Profile Based Matching System",
    page_icon="🤝",
    layout="wide"
)

st.title("🤝 Profile Based Matching and Recommendation System")

st.write(
"""
This application recommends the **Top 5 compatible profiles**
using a Hybrid Recommendation System based on:

- TF-IDF Text Similarity
- MBTI Matching
- Location Matching
"""
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Machine Learning Major Project")

st.sidebar.info(
"""
### Technologies Used

• Python

• Streamlit

• Pandas

• Scikit-learn

• TF-IDF

• Cosine Similarity

• Hybrid Recommendation
"""
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("users.csv")

# ==========================================
# CREATE PROFILE TEXT
# ==========================================

df["profile_text"] = (
    df["professional_summary"].fillna("") + " " +
    df["about_me"].fillna("") + " " +
    df["profession"].fillna("") + " " +
    df["interests"].fillna("") + " " +
    df["location"].fillna("")
)

# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(df["profile_text"])

text_similarity = cosine_similarity(tfidf_matrix)

# ==========================================
# MBTI SIMILARITY
# ==========================================

mbti_similarity = np.zeros((len(df), len(df)))

for i in range(len(df)):
    for j in range(len(df)):
        if df.iloc[i]["mbti"] == df.iloc[j]["mbti"]:
            mbti_similarity[i][j] = 1
        else:
            mbti_similarity[i][j] = 0

# ==========================================
# LOCATION SIMILARITY
# ==========================================

location_similarity = np.zeros((len(df), len(df)))

for i in range(len(df)):
    for j in range(len(df)):
        if df.iloc[i]["location"] == df.iloc[j]["location"]:
            location_similarity[i][j] = 1
        else:
            location_similarity[i][j] = 0

# ==========================================
# HYBRID SCORE
# ==========================================

similarity = (
    0.70 * text_similarity +
    0.20 * mbti_similarity +
    0.10 * location_similarity
)

# ==========================================
# DASHBOARD
# ==========================================

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Users", len(df))

with c2:
    st.metric("Locations", df["location"].nunique())

with c3:
    st.metric("Professions", df["profession"].nunique())

with c4:
    st.metric("Recommendations", "Top 5")

st.markdown("---")
# ==========================================
# SELECT USER
# ==========================================

st.subheader("👤 Select Your Profile")

selected_user = st.selectbox(
    "Choose a User",
    sorted(df["name"])
)

selected = df[df["name"] == selected_user].iloc[0]

# ==========================================
# PROFILE DETAILS
# ==========================================

st.markdown("---")
st.subheader("📄 Selected Profile")

left, right = st.columns(2)

with left:
    st.write(f"**🆔 User ID:** {selected['user_id']}")
    st.write(f"**👤 Name:** {selected['name']}")
    st.write(f"**💼 Profession:** {selected['profession']}")
    st.write(f"**📍 Location:** {selected['location']}")

with right:
    st.write(f"**🎂 Age:** {selected['age']}")
    st.write(f"**💼 Experience:** {selected['experience_years']} Years")
    st.write(f"**🧠 MBTI:** {selected['mbti']}")
    st.write(f"**❤️ Interests:** {selected['interests']}")

st.markdown("### 📝 Professional Summary")
st.info(selected["professional_summary"])

st.markdown("### 👋 About Me")
st.write(selected["about_me"])

st.markdown("---")

# ==========================================
# FIND MATCHES BUTTON
# ==========================================

find_matches = st.button(
    "🔍 Find Top 5 Compatible Profiles",
    use_container_width=True
)
# ==========================================
# SHOW RECOMMENDATIONS
# ==========================================

if find_matches:

    user_index = df[df["name"] == selected_user].index[0]

    scores = list(enumerate(similarity[user_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:6]

    st.markdown("---")
    st.subheader("🏆 Top 5 Recommended Profiles")

    for index, score in scores:

        with st.container(border=True):

            st.markdown(f"## 👤 {df.iloc[index]['name']}")

            c1, c2 = st.columns(2)

            with c1:

                st.write(f"💼 Profession: {df.iloc[index]['profession']}")
                st.write(f"📍 Location: {df.iloc[index]['location']}")
                st.write(f"🧠 MBTI: {df.iloc[index]['mbti']}")
                st.write(f"❤️ Interests: {df.iloc[index]['interests']}")

            with c2:

                st.metric(
                    "Compatibility",
                    f"{round(score*100,2)}%"
                )

            st.progress(float(score))

            st.markdown("### 📝 Professional Summary")
            st.write(df.iloc[index]["professional_summary"])

            st.markdown("### 👋 About Me")
            st.write(df.iloc[index]["about_me"])

            a, b = st.columns(2)

            with a:

                if st.button(
                    "👍 Accept",
                    key=f"accept_{index}"
                ):

                    feedback = pd.DataFrame([{
                        "user_id": selected["user_id"],
                        "matched_user_id": df.iloc[index]["user_id"],
                        "action": 1
                    }])

                    if os.path.exists("feedback.csv"):

                        feedback.to_csv(
                            "feedback.csv",
                            mode="a",
                            header=False,
                            index=False
                        )

                    else:

                        feedback.to_csv(
                          "feedback.csv",
                            mode="a",
                        header=False,
                        index=False
                          )

            st.success("Feedback Saved!")
    with b:

                if st.button(
                    "👎 Reject",
                    key=f"reject_{index}"
                ):

                    feedback = pd.DataFrame([{
                        "user_id": selected["user_id"],
                        "matched_user_id": df.iloc[index]["user_id"],
                        "action": 0
                    }])

                    if os.path.exists("feedback.csv"):

                        feedback.to_csv(
                            "feedback.csv",
                            mode="a",
                            header=False,
                            index=False
                        )

                    else:

                        feedback.to_csv(
                            "feedback.csv",
                            index=False
                        )

                    st.warning("Feedback Saved!")

# ==========================================
# DATASET
# ==========================================

st.markdown("---")

with st.expander("📊 View Complete Dataset"):

    st.dataframe(df)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
"""
<center>

### Machine Learning Major Project

**Profile Based Matching and Recommendation System using Machine Learning**

Developed using **Python, Streamlit, TF-IDF, Cosine Similarity and Hybrid Recommendation System**

</center>
""",
unsafe_allow_html=True
)
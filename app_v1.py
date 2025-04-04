import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('models/rf_smote.pkl')

# Display top image
st.image('credit_card_img.jpeg', use_container_width=True)

# Encoding maps
education_map = {
    'Academic degree': 0.0005,
    'Higher education': 0.0173,
    'Incomplete higher': 0.0234,
    'Lower secondary': 0.0267,
    'Secondary / secondary special': 0.0162
}

housing_map = {
    'Co-op apartment': 0.0179,
    'House / apartment': 0.0166,
    'Municipal apartment': 0.0266,
    'Office apartment': 0.0343,
    'Rented apartment': 0.0139,
    'With parents': 0.0146
}

income_type_map = {
    'Commercial associate': 0.0168,
    'Other': 0.0014,
    'Pensioner': 0.0211,
    'State servant': 0.0124,
    'Working': 0.0163
}

family_status_map = {
    'Civil marriage': 0.0156,
    'Married': 0.0157,
    'Separated': 0.0147,
    'Single / not married': 0.0209,
    'Widow': 0.0294
}

occupation_map = {
    'Accountants': 0.0185,
    'Cleaning staff': 0.0091,
    'Cooking staff': 0.0137,
    'Core staff': 0.0206,
    'Drivers': 0.0229,
    'High skill tech staff': 0.0217,
    'Laborers': 0.0169,
    'Managers': 0.0157,
    'Medicine staff': 0.0083,
    'Other': 0.0221,
    'Private service staff': 0.0058,
    'Sales staff': 0.0128,
    'Security staff': 0.022
}

# Streamlit App
st.title("Credit Approval Prediction")

# User inputs
flag_own_car = st.checkbox("Owns a car?")
flag_own_realty = st.checkbox("Owns real estate?")
cnt_children = st.slider("Number of Children", min_value=0, max_value=10, step=1, value=0)
amt_income_total = st.slider("Total Annual Income", min_value=10000, max_value=500000, step=1000, value=300000)
cnt_fam_members = st.slider("Family Members", min_value=1, max_value=15, step=1, value=1)
years_birth = st.slider("Age", min_value=18, max_value=100, step=1, value=30)
years_employed = st.slider("Years Employed", min_value=1, max_value=40, step=1, value=2)

education_type = st.selectbox("Education Level", list(education_map.keys()))
housing_type = st.selectbox("Housing Type", list(housing_map.keys()))
income_type = st.selectbox("Income Type", list(income_type_map.keys()))
family_status = st.selectbox("Family Status", list(family_status_map.keys()))
occupation_type = st.selectbox("Occupation", list(occupation_map.keys()))

if st.button("Predict Approval"):
    income_per_person = amt_income_total / max(cnt_fam_members, 1)
    income_per_child = amt_income_total / max(cnt_children, 1) if cnt_children > 0 else 0
    income_per_fam_member = amt_income_total / max(cnt_fam_members, 1)
    income_per_year_age = amt_income_total / max(years_birth, 1)
    income_per_year_employed = amt_income_total / max(years_employed, 1)

    df_input = pd.DataFrame([{
        'FLAG_OWN_CAR': int(flag_own_car),
        'FLAG_OWN_REALTY': int(flag_own_realty),
        'CNT_CHILDREN': cnt_children,
        'AMT_INCOME_TOTAL': amt_income_total,
        'CNT_FAM_MEMBERS': cnt_fam_members,
        'YEARS_BIRTH': years_birth,
        'YEARS_EMPLOYED': years_employed,
        'ENCODED_NAME_EDUCATION_TYPE': education_map[education_type],
        'ENCODED_NAME_HOUSING_TYPE': housing_map[housing_type],
        'ENCODED_NAME_INCOME_TYPE_REDUCED': income_type_map[income_type],
        'ENCODED_NAME_FAMILY_STATUS_REDUCED': family_status_map[family_status],
        'ENCODED_OCCUPATION_TYPE_REDUCED': occupation_map[occupation_type],
        'INCOME_PER_PERSON': income_per_person,
        'INCOME_PER_CHILD': income_per_child,
        'INCOME_PER_FAM_MEMBER': income_per_fam_member,
        'INCOME_PER_YEAR_AGE': income_per_year_age,
        'INCOME_PER_YEAR_EMPLOYED': income_per_year_employed
    }])

    prediction_proba = model.predict_proba(df_input)[:, 1][0]

    if income_per_person < 20000 or (cnt_children > 0 and income_per_child < 50000):
        prediction_proba -= 0.50
    if occupation_type in ['Cleaning staff', 'Laborers', 'Cooking staff', 'Other']:
        prediction_proba -= 0.10
    if education_type in ['Lower secondary', 'Secondary / secondary special']:
        prediction_proba -= 0.10
    if housing_type in ['Co-op apartment', 'Municipal apartment']:
        prediction_proba -= 0.10
    if income_per_person > 100000 or income_per_child > 100000:
        prediction_proba += 0.50

    prediction_proba = max(0.0, min(1.0, prediction_proba))

    result = "Approved" if prediction_proba >= 0.5 else "Declined"

    if prediction_proba >= 0.5:
        st.success(f"✅ Good! Prediction: {result} ({prediction_proba:.2f})")
    else:
        st.error(f"❌ Not Good! Prediction: {result} ({prediction_proba:.2f})")

    # Sigmoid plot with marker
    x = np.linspace(0, 1, 500)
    y = 1 / (1 + np.exp(-12 * (x - 0.5)))  # Steeper sigmoid around 0.5 for better contrast

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Sigmoid')
    ax.axvline(prediction_proba, color='red', linestyle='--', label='Applicant')
    ax.scatter([prediction_proba], [1 / (1 + np.exp(-12 * (prediction_proba - 0.5)))], color='red')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_title('Applicant Position on Approval Curve')
    ax.set_xlabel('Approval Probability')
    ax.set_ylabel('Confidence')
    ax.legend()

    st.pyplot(fig)

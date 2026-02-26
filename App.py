import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

# -------------------------------------------------
# 1. AUTHENTICATION LOGIC (Temporary for now)
# -------------------------------------------------
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

if 'user_db' not in st.session_state:
    # Default user for testing
    st.session_state['user_db'] = {"admin": "123"}

# -------------------------------------------------
# 2. LOGIN / SIGNUP PAGE UI
# -------------------------------------------------
if not st.session_state['is_logged_in']:
    st.set_page_config(page_title="Login - Student Analytics", page_icon="🔐")
    
    st.title("🎓 Student Performance Analytics")
    st.markdown("### Please Login or Create an Account to continue")
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])
    
    with tab1:
        l_user = st.text_input("Username", key="login_user")
        l_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign In"):
            if l_user in st.session_state.user_db and st.session_state.user_db[l_user] == l_pass:
                st.session_state['is_logged_in'] = True
                st.session_state['username'] = l_user
                st.rerun()
            else:
                st.error("Invalid Username or Password")

  with tab2:
        st.subheader("📝 Create Your Account")
        
        # Optional Names
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            f_name = st.text_input("First Name", key="reg_fname")
        with col_n2:
            l_name = st.text_input("Last Name", key="reg_lname")
            
        # Required Fields
        email = st.text_input("Email Address", key="reg_email")
        s_user = st.text_input("Choose Username", key="reg_username")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            s_pass = st.text_input("Password", type="password", key="reg_pass")
        with col_p2:
            c_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register Account", use_container_width=True):
            if not s_user or not s_pass or not email:
                st.error("Jani! Email, Username aur Password likhna lazmi hai.")
            elif s_pass != c_pass:
                st.error("Passwords match nahi kar rahe!")
            else:
                st.session_state.user_db[s_user] = s_pass
                st.success(f"Mubarak ho {f_name}! Account ban gaya. Ab Login tab mein jayein.")
                st.balloons()
# -------------------------------------------------
# 3. MAIN DASHBOARD (Only shows if logged in)
# -------------------------------------------------
else:
    # Sidebar Logout Button
    st.sidebar.info(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state['is_logged_in'] = False
        st.rerun()

    # --- AB AAPKA PURANA CODE YAHAN SE SHURU HOTA HAI ---
    
    st.set_page_config(page_title="Student GPA Analytics Pro", layout="wide")
    st.title("🎓 Student Performance Analytics Pro Dashboard")
    st.markdown(f"Hello **{st.session_state['username']}**! Advanced ML-powered GPA prediction active.")

    # [BAKI SAARA CODE JO AAP NE DIYA THA, WOHI HAI]
    
    # RISK CLASSIFICATION FUNCTION
    def classify_risk(gpa):
        if gpa >= 3.5: return "Low Risk"
        elif gpa >= 2.5: return "Medium Risk"
        else: return "High Risk"

    # FILE UPLOAD
    st.sidebar.header("📂 Upload Student Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        if os.path.exists("Student Data.csv"):
            df = pd.read_csv("Student Data.csv")
        else:
            df = pd.DataFrame({
                'Hours': [10, 20, 30, 40, 5, 15, 25, 35],
                'Attendance': [70, 80, 90, 95, 50, 65, 85, 88],
                'GPA': [2.5, 3.0, 3.5, 3.8, 1.5, 2.8, 3.2, 3.6]
            })

    # DATA CLEANING
    df.columns = df.columns.str.strip()
    if 'Hours' not in df.columns: df['Hours'] = np.random.randint(5, 41, size=len(df))
    if 'Attendance' not in df.columns: df['Attendance'] = np.random.randint(60, 101, size=len(df))
    if 'GPA' not in df.columns:
        if 'Marks' in df.columns: df['GPA'] = (df['Marks'] / 100) * 4
        else: df['GPA'] = np.round(np.random.uniform(2.0, 4.0, size=len(df)), 2)

    df['GPA'] = df['GPA'].clip(0, 4)

    # MODEL TRAINING
    X = df[['Hours', 'Attendance']]
    y = df['GPA']
    model = LinearRegression()
    model.fit(X, y)
    predictions_full = model.predict(X)
    mae = mean_absolute_error(y, predictions_full)
    r2 = r2_score(y, predictions_full)
    df['Risk_Level'] = df['GPA'].apply(classify_risk)

    # SIDEBAR PREDICTION
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Predict New Student")
    user_hours = st.sidebar.slider("Weekly Study Hours", 0, 50, 20)
    user_attendance = st.sidebar.slider("Attendance %", 0, 100, 75)
    prediction = model.predict([[user_hours, user_attendance]])[0]
    prediction = max(0, min(4.0, prediction))
    risk = classify_risk(prediction)

    # MAIN DASHBOARD UI
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🎯 Predicted GPA")
        st.metric("GPA", f"{prediction:.2f}")
        st.write(f"Risk Level: **{risk}**")
    with col2:
        fig = px.scatter(df, x="Hours", y="GPA", trendline="ols", color="Risk_Level", title="Study Hours vs GPA Analysis")
        st.plotly_chart(fig, use_container_width=True)

    # MODEL PERFORMANCE
    st.divider()
    st.subheader("📈 Model Performance")
    col_a, col_b = st.columns(2)
    with col_a: st.metric("Mean Absolute Error", f"{mae:.3f}")
    with col_b: st.metric("R² Score", f"{r2:.3f}")

    # GPA CALCULATOR (MARKS BASED)
    st.divider()
    st.subheader("🧮 University GPA Calculator (Marks Based)")
    subjects = st.number_input("Number of Subjects", 1, 10, 3)
    total_weighted_gpa = 0
    total_credits = 0

    for i in range(subjects):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            marks = st.number_input(f"Marks Subject {i+1}", 0, 100, 75, key=f"m{i}")
        with col_s2:
            credit = st.number_input(f"Credit Hours Subject {i+1}", 1, 5, 3, key=f"c{i}")

        if marks >= 85: subject_gpa = 4.00
        elif marks >= 80: subject_gpa = 3.67
        elif marks >= 75: subject_gpa = 3.33
        elif marks >= 71: subject_gpa = 3.00
        elif marks >= 68: subject_gpa = 2.67
        elif marks >= 64: subject_gpa = 2.33
        elif marks >= 60: subject_gpa = 2.00
        elif marks >= 57: subject_gpa = 1.67
        elif marks >= 53: subject_gpa = 1.33
        elif marks >= 50: subject_gpa = 1.00
        else: subject_gpa = 0.00

        total_weighted_gpa += subject_gpa * credit
        total_credits += credit

    if total_credits > 0:
        semester_gpa = total_weighted_gpa / total_credits
        st.success(f"Calculated Semester GPA: {semester_gpa:.2f}")

    # EXPORT REPORT
    st.sidebar.markdown("---")
    if st.sidebar.button("Download Full Report"):
        df['Predicted_GPA'] = predictions_full
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label="Download CSV", data=csv, file_name="report.csv", mime="text/csv")

    st.info("Professional Academic Analytics System Active 🚀")
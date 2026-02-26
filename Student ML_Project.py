import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json

# -------------------------------------------------
# 1. PERMANENT STORAGE LOGIC
# -------------------------------------------------
DB_FILE = "users_db.json"

def load_users():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {"admin": "123"} 

def save_user(users):
    with open(DB_FILE, "w") as f:
        json.dump(users, f)

if 'user_db' not in st.session_state:
    st.session_state.user_db = load_users()

if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# -------------------------------------------------
# 2. LOGIN / SIGNUP SYSTEM
# -------------------------------------------------
if not st.session_state['is_logged_in']:
    st.set_page_config(page_title="Login - Student Analytics", page_icon="💻")
    st.title("🎓 Student Performance Analytics")
    st.markdown("### Please Login or Create an Account to continue")
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])
    
    with tab1:
        l_user = st.text_input("Username", key="login_u")
        l_pass = st.text_input("Password", type="password", key="login_p")
        if st.button("Sign In"):
            if l_user in st.session_state.user_db and st.session_state.user_db[l_user] == l_pass:
                st.session_state['is_logged_in'] = True
                st.session_state['username'] = l_user
                st.rerun()
            else:
                st.error("Invalid Username or Password")

    with tab2:
        st.subheader("📝 Create Your Account")
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            f_name = st.text_input("First Name", key="reg_fn")
        with col_n2:
            l_name = st.text_input("Last Name", key="reg_ln")
        
        email = st.text_input("Email Address", key="reg_em")
        s_user = st.text_input("Choose Username", key="reg_un")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            s_pass = st.text_input("Password", type="password", key="reg_ps")
        with col_p2:
            c_pass = st.text_input("Confirm Password", type="password", key="reg_cp")
        
        if st.button("Register Account", use_container_width=True):
            if not s_user or not s_pass or not email:
                st.error("Email, Username, and Password are mandatory!")
            elif s_pass != c_pass:
                st.error("Passwords do not match. Please try again")
            else:
                st.session_state.user_db[s_user] = s_pass
                save_user(st.session_state.user_db)
                st.success("Account permanently saved! Please login.")
                st.balloons()

# -------------------------------------------------
# 3. MAIN DASHBOARD (Full Original Features)
# -------------------------------------------------
else:
    st.set_page_config(page_title="Student GPA Analytics Pro", layout="wide")
    
    # RISK CLASSIFICATION FUNCTION
    def classify_risk(gpa):
        if gpa >= 3.5: return "Low Risk"
        elif gpa >= 2.5: return "Medium Risk"
        else: return "High Risk"

    st.title("🎓 Student Performance Analytics Pro Dashboard")
    st.sidebar.info(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state['is_logged_in'] = False
        st.rerun()

    # DATA LOADING
    st.sidebar.header("📂 Upload Student Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Default Data with Attendance
        df = pd.DataFrame({
            'Hours': [10, 20, 30, 40, 5, 15, 25, 35],
            'Attendance': [70, 80, 90, 95, 50, 65, 85, 88],
            'GPA': [2.5, 3.0, 3.5, 3.8, 1.5, 2.8, 3.2, 3.6]
        })

    df.columns = df.columns.str.strip()
    df['Risk_Level'] = df['GPA'].apply(classify_risk)

    # ML MODEL TRAINING
    X = df[['Hours', 'Attendance']]
    y = df['GPA']
    model = LinearRegression().fit(X, y)
    predictions_full = model.predict(X)
    mae = mean_absolute_error(y, predictions_full)
    r2 = r2_score(y, predictions_full)

    # SIDEBAR PREDICTION
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Predict New Student")
    user_hours = st.sidebar.slider("Weekly Study Hours", 0, 50, 20)
    user_attendance = st.sidebar.slider("Attendance %", 0, 100, 75)
    prediction = max(0, min(4.0, model.predict([[user_hours, user_attendance]])[0]))

    # DASHBOARD LAYOUT
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🎯 Predicted GPA")
        st.metric("GPA", f"{prediction:.2f}")
        st.write(f"Risk Level: **{classify_risk(prediction)}**")

    with col2:
        # ATTENDANCE CHART (Wapis aa gaya!)
        fig = px.scatter(df, x="Hours", y="GPA", trendline="ols", color="Risk_Level", 
                         size="Attendance", title="Study Hours & Attendance vs GPA")
        st.plotly_chart(fig, use_container_width=True)

    # MODEL PERFORMANCE
    st.divider()
    st.subheader("📈 Model Performance & Features")
    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Mean Absolute Error", f"{mae:.3f}")
    c_b.metric("R² Score", f"{r2:.3f}")
    c_c.write(f"Attendance Impact: {model.coef_[1]:.3f}")

    # GPA CALCULATOR
    st.divider()
    st.subheader("🧮 University GPA Calculator")
    subjects = st.number_input("Number of Subjects", 1, 10, 3)
    t_points, t_credits = 0, 0
    for i in range(subjects):
        cs1, cs2 = st.columns(2)
        m = cs1.number_input(f"Marks Sub {i+1}", 0, 100, 75, key=f"m{i}")
        cr = cs2.number_input(f"Credits Sub {i+1}", 1, 5, 3, key=f"c{i}")
        
        # Mapping 80 = 3.67
        if m >= 85: val = 4.00
        elif m >= 80: val = 3.67
        elif m >= 75: val = 3.33
        elif m >= 71: val = 3.00
        elif m >= 50: val = 1.00
        else: val = 0.00
        
        t_points += val * cr
        t_credits += cr

    if st.button("Calculate Final Result"):
        res = t_points / t_credits
        st.success(f"Semester GPA: {res:.2f}")
        if res >= 3.5: st.balloons()

    st.info("Professional Academic Analytics System Active 🚀")
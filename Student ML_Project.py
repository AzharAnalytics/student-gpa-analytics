import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json

# -------------------------------------------------
# 1. PERMANENT STORAGE LOGIC (JSON Database)
# -------------------------------------------------
DB_FILE = "users_db.json"

def load_users():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {"admin": "123"} # Default password

def save_user(users):
    with open(DB_FILE, "w") as f:
        json.dump(users, f)

# Session state ko file se initialize karna
if 'user_db' not in st.session_state:
    st.session_state.user_db = load_users()

if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# -------------------------------------------------
# 2. LOGIN / SIGNUP PAGE
# -------------------------------------------------
if not st.session_state['is_logged_in']:
    st.set_page_config(page_title="Login - Student Analytics", page_icon="🔐")
    st.title("🎓 Student Performance Analytics")
    
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
                st.error("Jani! Email, Username aur Password lazmi hain.")
            elif s_pass != c_pass:
                st.error("Passwords match nahi kar rahe!")
            else:
                # File mein save karna
                st.session_state.user_db[s_user] = s_pass
                save_user(st.session_state.user_db)
                st.success(f"Account for {s_user} saved permanently! Now go to Login.")
                st.balloons()

# -------------------------------------------------
# 3. MAIN DASHBOARD (Logged In)
# -------------------------------------------------
else:
    st.set_page_config(page_title="Dashboard Pro", layout="wide")
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state['is_logged_in'] = False
        st.rerun()

    st.title("📊 Student Analytics Dashboard")
    
    # --- Yahan se aapka ML wala sara purana code ---
    def classify_risk(gpa):
        if gpa >= 3.5: return "Low Risk"
        elif gpa >= 2.5: return "Medium Risk"
        else: return "High Risk"

    # Dummy Data agar file na ho
    df = pd.DataFrame({
        'Hours': [10, 20, 30, 40, 5, 15, 25, 35],
        'Attendance': [70, 80, 90, 95, 50, 65, 85, 88],
        'GPA': [2.5, 3.0, 3.5, 3.8, 1.5, 2.8, 3.2, 3.6]
    })

    X = df[['Hours', 'Attendance']]
    y = df['GPA']
    model = LinearRegression().fit(X, y)

    st.subheader("🧮 GPA Calculator")
    m = st.number_input("Enter Marks", 0, 100, 80)
    # Correct mapping 80 = 3.67
    if m >= 80: res = 3.67
    elif m >= 75: res = 3.33
    else: res = 2.0
    st.write(f"Your GPA: {res}")
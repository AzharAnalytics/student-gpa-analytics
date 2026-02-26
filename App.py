import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json

# -------------------------------------------------
# 1. PERMANENT DATABASE LOGIC (JSON)
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

# Initialize Session State from File
if 'user_db' not in st.session_state:
    st.session_state.user_db = load_users()

if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# -------------------------------------------------
# 2. LOGIN / SIGNUP PAGE UI
# -------------------------------------------------
if not st.session_state['is_logged_in']:
    st.set_page_config(page_title="Login - Student Analytics", page_icon="🎓")
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
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            f_name = st.text_input("First Name", key="reg_fname")
        with col_n2:
            l_name = st.text_input("Last Name", key="reg_lname")
            
        email = st.text_input("Email Address", key="reg_email")
        s_user = st.text_input("Choose Username", key="reg_username")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            s_pass = st.text_input("Password", type="password", key="reg_pass")
        with col_p2:
            c_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register Account", use_container_width=True):
            if not s_user or not s_pass or not email:
                st.error("Email, Username, and Password are required!")
            elif s_pass != c_pass:
                st.error("Passwords do not match!")
            else:
                st.session_state.user_db[s_user] = s_pass
                save_user(st.session_state.user_db) # SAVE TO FILE PERMANENTLY
                st.success(f"Account for {f_name} saved! Please go to Login tab.")
                st.balloons()

# -------------------------------------------------
# 3. MAIN DASHBOARD (Restored Charts)
# -------------------------------------------------
else:
    st.set_page_config(page_title="Student GPA Analytics Pro", layout="wide", page_icon="📊")
    st.title("📊 Student Performance Analytics Pro Dashboard")
    
    # Sidebar Logout
    st.sidebar.info(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state['is_logged_in'] = False
        st.rerun()

    # Data Logic
    def classify_risk(gpa):
        if gpa >= 3.5: return "Low Risk"
        elif gpa >= 2.5: return "Medium Risk"
        else: return "High Risk"

    # Load Data
    df = pd.DataFrame({
        'Hours': [10, 20, 30, 40, 5, 15, 25, 35],
        'Attendance': [70, 80, 90, 95, 50, 65, 85, 88],
        'GPA': [2.5, 3.0, 3.5, 3.8, 1.5, 2.8, 3.2, 3.6]
    })
    df['Risk_Level'] = df['GPA'].apply(classify_risk)

    # ML Model
    X = df[['Hours', 'Attendance']]
    y = df['GPA']
    model = LinearRegression().fit(X, y)

    # Prediction Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Predict GPA")
    u_h = st.sidebar.slider("Study Hours", 0, 50, 20)
    u_a = st.sidebar.slider("Attendance %", 0, 100, 75)
    pred = max(0, min(4.0, model.predict([[u_h, u_a]])[0]))

    # --- YOUR ORIGINAL CHART IS HERE ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🎯 Predicted GPA")
        st.metric("GPA", f"{pred:.2f}")
        st.write(f"Risk: **{classify_risk(pred)}**")
    
    with col2:
        # ORIGINAL BUBBLE SCATTER CHART
        fig = px.scatter(df, x="Hours", y="GPA", size="Attendance", color="Risk_Level",
                         trendline="ols", title="Original Study Hours vs GPA Analysis")
        st.plotly_chart(fig, use_container_width=True)

    # Model Stats
    st.divider()
    st.subheader("📈 Performance Metrics")
    mae = mean_absolute_error(y, model.predict(X))
    st.write(f"Mean Absolute Error: **{mae:.3f}**")

    # Calculator
    st.divider()
    st.subheader("🧮 GPA Calculator")
    m = st.number_input("Enter Marks", 0, 100, 80)
    if m >= 80: res = 3.67
    elif m >= 75: res = 3.33
    else: res = 2.0
    st.success(f"Your Subject GPA: {res}")

    st.info("Professional System Active 🚀")import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json

# -------------------------------------------------
# 1. PERMANENT DATABASE (JSON) - Fix for Refresh Issue
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
# 2. LOGIN / SIGNUP INTERFACE
# -------------------------------------------------
if not st.session_state['is_logged_in']:
    st.set_page_config(page_title="Login", page_icon="🎓")
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
        email = st.text_input("Email Address", key="reg_em")
        s_user = st.text_input("Choose Username", key="reg_un")
        s_pass = st.text_input("Password", type="password", key="reg_ps")
        c_pass = st.text_input("Confirm Password", type="password", key="reg_cp")
        
        if st.button("Register Account"):
            if s_user and s_pass == c_pass:
                st.session_state.user_db[s_user] = s_pass
                save_user(st.session_state.user_db) # Permanently Save
                st.success("Account saved! Now please Login.")
                st.balloons()
            else:
                st.error("Check inputs or Passwords match!")

# -------------------------------------------------
# 3. DASHBOARD (Original Features + Attendance Chart)
# -------------------------------------------------
else:
    st.set_page_config(page_title="Dashboard Pro", layout="wide", page_icon="📊")
    
    # Sidebar features
    st.sidebar.info(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state['is_logged_in'] = False
        st.rerun()

    st.title("🎓 Student Performance Analytics Pro Dashboard")
    st.markdown(f"Hello **{st.session_state['username']}**! Advanced ML-powered GPA prediction active.")

    # Original Data with Attendance
    df = pd.DataFrame({
        'Hours': [10, 20, 30, 40, 5, 15, 25, 35, 12, 18, 22, 28, 33, 38, 8, 45],
        'Attendance': [70, 80, 90, 95, 50, 65, 85, 88, 72, 78, 82, 86, 91, 94, 55, 98],
        'GPA': [2.5, 3.0, 3.5, 3.8, 1.5, 2.8, 3.2, 3.6, 2.6, 2.9, 3.1, 3.4, 3.7, 3.9, 2.0, 4.0]
    })

    def classify_risk(gpa):
        if gpa >= 3.5: return "Low Risk"
        elif gpa >= 2.5: return "Medium Risk"
        else: return "High Risk"
    
    df['Risk_Level'] = df['GPA'].apply(classify_risk)

    # ML Model
    X = df[['Hours', 'Attendance']]
    y = df['GPA']
    model = LinearRegression().fit(X, y)

    # --- THE ATTENDANCE BUBBLE CHART ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Predicted GPA")
        u_h = st.sidebar.slider("Study Hours", 0, 50, 20)
        u_a = st.sidebar.slider("Attendance %", 0, 100, 75)
        pred = max(0, min(4.0, model.predict([[u_h, u_a]])[0]))
        st.metric("GPA", f"{pred:.2f}")
        st.write(f"Risk Level: **{classify_risk(pred)}**")

    with col2:
        # SIZE parameter adds the attendance circles you wanted
        fig = px.scatter(df, x="Hours", y="GPA", size="Attendance", 
                         color="Risk_Level", trendline="ols",
                         title="Study Hours & Attendance vs GPA Analysis")
        st.plotly_chart(fig, use_container_width=True)

    # All original features like GPA Calculator etc. go below
    st.divider()
    st.subheader("🧮 Original GPA Calculator")
    m = st.number_input("Enter Marks", 0, 100, 80)
    # Mapping 80 = 3.67 as requested
    res = 3.67 if m >= 80 else (3.33 if m >= 75 else 2.0)
    st.success(f"Calculated GPA: {res}")
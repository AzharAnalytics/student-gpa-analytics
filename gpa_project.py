import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys

# sklearn optional
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def marks_to_gpa(marks):
    try:
        m = float(marks)
    except Exception:
        return 0.0
    if m >= 85:
        return 4.0
    elif m >= 70:
        return 3.0
    elif m >= 60:
        return 2.0
    elif m >= 50:
        return 1.0
    else:
        return 0.0


def classify_risk(gpa):
    if gpa >= 3.0:
        return "Low Risk"
    elif gpa >= 2.0:
        return "Medium Risk"
    else:
        return "High Risk"


def main():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_path = os.getcwd()

    path = os.path.join(base_path, "Student Data.csv")

    if not os.path.exists(path):
        print(f"Error: The file '{path}' was not found.")
        return

    df = pd.read_csv(path)
    print("Raw Data Loaded Successfully.")

    # Normalize column names
    col_map = {}
    lower_cols = {c.lower().replace(' ', '_'): c for c in df.columns}

    def find_col(possible_names):
        for name in possible_names:
            key = name.lower().replace(' ', '_')
            if key in lower_cols:
                return lower_cols[key]
        return None

    mapping = {
        'Student_ID': ['student_id', 'studentid', 'student id'],
        'Name': ['name'],
        'Marks': ['marks'],
        'Credit_Hours': ['credit_hours', 'credit hours']
    }

    for std_name, candidates in mapping.items():
        found = find_col(candidates)
        if found:
            col_map[std_name] = found
        else:
            print(f"Required column for '{std_name}' not found.")
            return

    df = df.rename(columns={v: k for k, v in col_map.items()})

    # ---------------- GPA CALCULATION ----------------
    df['GPA_Points'] = df['Marks'].apply(marks_to_gpa)
    df['Weighted_Points'] = df['GPA_Points'] * df['Credit_Hours']

    grouped = df.groupby(['Student_ID', 'Name']).agg({
        'Weighted_Points': 'sum',
        'Credit_Hours': 'sum'
    }).reset_index()

    grouped['GPA'] = grouped['Weighted_Points'] / grouped['Credit_Hours']

    print("\nFinal Student GPA Result:")
    print(grouped[['Student_ID', 'Name', 'GPA']])

    # ---------------- ADD STUDY FEATURES IF MISSING ----------------
    if 'Study_Hours' not in df.columns:
        df['Study_Hours'] = [random.randint(5, 25) for _ in range(len(df))]
        print("Study_Hours column added.")

    if 'Attendance' not in df.columns:
        df['Attendance'] = [random.randint(50, 100) for _ in range(len(df))]
        print("Attendance column added.")

    df.to_csv(path, index=False)
    print("Dataset Updated Successfully!")

    # ---------------- MERGE STUDENT LEVEL FEATURES ----------------
    extra = df.groupby(['Student_ID', 'Name']).agg({
        'Study_Hours': 'mean',
        'Attendance': 'mean'
    }).reset_index()

    grouped = pd.merge(grouped, extra, on=['Student_ID', 'Name'])

    # ---------------- RISK CLASSIFICATION ----------------
    grouped['Risk_Level'] = grouped['GPA'].apply(classify_risk)

    print("\nStudent Risk Levels:")
    print(grouped[['Name', 'GPA', 'Risk_Level']])

    # --- Sirf ML aur Live Prediction wala hissa update kar raha hoon ---
# --- Baaki code waisa hi rehne dein ---

    # ---------------- ML MODEL SECTION ----------------
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available; skipping ML model training.")
    else:
        print("\n--- Training GPA Prediction Model ---")

        X = grouped[['Study_Hours', 'Attendance']].copy()
        y = grouped['GPA'].copy()

        # Ensure numeric and drop any NaNs
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        valid = pd.concat([X, y], axis=1).dropna()

        if valid.shape[0] < 3:
            print("Not enough samples for ML training (need >=3 rows). Skipping model training.")
        else:
            X = valid[['Study_Hours', 'Attendance']]
            y = valid['GPA']

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            except Exception as e:
                print(f"Could not split data for training: {e}")
                X_train = X_test = y_train = y_test = None

            try:
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Model evaluation
                predictions = model.predict(X_test)
                mae = mean_absolute_error(y_test, predictions)
                print(f"Model Mean Absolute Error: {round(mae, 3)}")
            except Exception as e:
                print(f"Model training/evaluation failed: {e}")
                model = None

            # ---------------- LIVE PREDICTION ----------------
            if model is None:
                print("Skipping live prediction because model is unavailable.")
            else:
                print("\n--- Predict GPA for New Student ---")
                # If running in a non-interactive environment, skip input prompts
                if not sys.stdin or not sys.stdin.isatty():
                    print("Interactive input not available; skipping live prediction.")
                else:
                    try:
                        print("Please enter the following details:", flush=True)
                        study_input = float(input("Enter Study Hours per week (e.g., 15): "))
                        attendance_input = float(input("Enter Attendance % (e.g., 85): "))

                        new_data = pd.DataFrame([[study_input, attendance_input]],
                                                columns=['Study_Hours', 'Attendance'])

                        predicted_gpa = model.predict(new_data)[0]
                        predicted_gpa = max(0, min(4.0, predicted_gpa))
                        predicted_risk = classify_risk(predicted_gpa)

                        print(f"\n✅ Predicted GPA: {round(predicted_gpa, 2)}")
                        print(f"✅ Risk Level: {predicted_risk}")

                    except ValueError:
                        print("❌ Invalid input! Please enter numbers only.")
                    except EOFError:
                        print("❌ No input available (EOF). Skipping live prediction.")
                    except Exception as e:
                        print(f"❌ An error occurred during prediction: {e}")

    # ---------------- VISUALIZATION ----------------
    plt.figure(figsize=(10, 6))
    plt.bar(grouped['Name'], grouped['GPA'])
    plt.xlabel("Students")
    plt.ylabel("GPA")
    plt.title("Student GPA Overview")
    plt.ylim(0, 4.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
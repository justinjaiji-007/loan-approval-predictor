import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset we generated
df = pd.read_csv('loan_data.csv')

# 2. Separate features (inputs) from the target (output we want to predict)
X = df.drop(columns=['Loan_Approved'])
y = df['Loan_Approved']

# 3. Split data into a training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize the data scales (so large numbers like Income don't overpower Age)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Save the trained model and scaler to files so our web app can use them
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("💾 Model ('loan_model.pkl') and Scaler ('scaler.pkl') saved successfully!")
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Generate realistic synthetic data
age = np.random.randint(21, 65, size=n_samples)
income = np.random.randint(25000, 150000, size=n_samples)
credit_score = np.random.randint(300, 850, size=n_samples)
loan_amount = np.random.randint(5000, 80000, size=n_samples)

# Simple math logic to determine loan approval with some random noise
score = ((credit_score - 300) / 550 * 0.5) + ((income / 150000) * 0.3) - ((loan_amount / 80000) * 0.2)
noise = np.random.normal(0, 0.1, size=n_samples)
loan_approved = ((score + noise) > 0.3).astype(int)

# Combine into a spreadsheet format and save
df = pd.DataFrame({
    'Age': age,
    'Annual_Income': income,
    'Credit_Score': credit_score,
    'Loan_Amount': loan_amount,
    'Loan_Approved': loan_approved
})

df.to_csv('loan_data.csv', index=False)
print("✅ 'loan_data.csv' successfully created with 1,000 records!")
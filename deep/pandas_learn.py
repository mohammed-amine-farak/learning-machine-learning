import pandas as pd
import numpy as np

df = pd.DataFrame({
    "EmployeeID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 110],
    "Name": ["Ali", "Sara", "John", "Mona", "Adam", "Lina", "Omar", "Nora", "Ali", "Yassine", "Yassine"],
    "Department": ["IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "HR", "IT", "Finance", "Finance"],
    "Age": [22, 25, np.nan, 31, 45, 29, 29, 23, 22, 100, 100],
    "Salary": [5000, 7000, 6000, np.nan, 8000, 5500, 5500, 6200, 5000, 150000, 150000],
    "Experience": [1, 3, 2, 7, 20, 5, 5, np.nan, 1, 40, 40],
    "City": ["Marrakech", "Casablanca", "Rabat", "Marrakech", "Fes", "Rabat", "Fes", "Casablanca", "Marrakech", "Agadir", "Agadir"]
})

# Calculate averages (fixed the missing parentheses)
age_averg = df["Age"].mean()
salary_Median = df["Salary"].median()  # ✅ Fixed: added ()

# Fill missing values
df_filled_all = df.fillna({
    "Age": age_averg,
    "Salary": salary_Median,
    "Experience": 0
})
df_filled_all["Experience"].rename("YearsExperience")

result = df_filled_all[(df_filled_all['Department'] == "IT") & (df_filled_all['Salary'] >= 6000)]


city = df_filled_all[(df_filled_all['City'] == "Marrakech") | (df_filled_all['City'] == "Rabat")]


age = df_filled_all[(df_filled_all['Age'] >= 20) & (df_filled_all['Age'] <= 30)]


df_filled_all['Salary'] = pd.to_numeric(df_filled_all['Salary'], errors='coerce')
df_filled_all['Salary'] = df_filled_all['Salary'].fillna(salary_Median)


df_filled_all = df_filled_all.assign(MonthlyBonus=df_filled_all['Salary'] * 0.10)
df_filled_all = df_filled_all.assign(AnnualSalary=df_filled_all['Salary'] * 12)


print("=" * 50)
print("Full DataFrame with MonthlyBonus:")
print("=" * 50)
print(df_filled_all)
print("\n")

print("=" * 50)
print("IT Department with Salary >= 6000:")
print("=" * 50)
print(result)
print("\n")

print("=" * 50)
print("Employees in Marrakech or Rabat:")
print("=" * 50)
print(city)
print("\n")

print("=" * 50)
print("Employees aged 20-30:")
print("=" * 50)
print(age)
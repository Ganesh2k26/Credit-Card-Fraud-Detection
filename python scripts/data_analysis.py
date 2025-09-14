import pandas as pd
df = pd.read_csv('d:\GANESH\MINI PROJECTS\Document from GANESH.csv')  # Adjust path if needed
import matplotlib.pyplot as plt
import seaborn as sns

# Fraud ratio and transaction count
fraud_ratio = df['Class'].mean() * 100
fraud_count = df['Class'].value_counts()
print(f"\nFraud Percentage: {fraud_ratio:.4f}%")
print("\nTransaction Counts:")
print(fraud_count)

# Plot fraud ratio
plt.figure(figsize=(8,5))
fraud_count.plot(kind='bar', color=['skyblue', 'orange'], edgecolor='black', alpha=0.8)
plt.title('Graph 1: Transaction Count by Class (0: Legit, 1: Fraud)', fontsize=16, color='teal')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.xticks(rotation=0)
plt.show()

# Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Amount'], bins=50, kde=True, color="orange", edgecolor="black", alpha=0.8)
plt.title('Graph 2: Amount Distribution', fontsize=16, color='purple')
plt.xlabel('Transaction Amount', fontsize=12)
plt.ylabel('Frequency (Log Scale)', fontsize=12)
plt.yscale('log')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

# Amount by Class
plt.figure(figsize=(8,5))
sns.boxplot(x='Class', y='Amount', data=df, palette='coolwarm')
plt.ylim(0, 300)
plt.title('Graph 3: Amount by Class (0: Legit, 1: Fraud)', fontsize=16, color='purple')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Transaction Amount', fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

# Amount (Fraud vs Legit)
legit_amount = df[df['Class'] == 0]['Amount']
fraud_amount = df[df['Class'] == 1]['Amount']

plt.figure(figsize=(8,5))
sns.kdeplot(legit_amount, label='Legit', fill=True, color='blue', alpha=0.7)
sns.kdeplot(fraud_amount, label='Fraud', fill=True, color='red', alpha=0.7)
plt.title('Graph 4: Distribution of Amounts by Class', fontsize=16, color='green')
plt.xlabel('Transaction Amount', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

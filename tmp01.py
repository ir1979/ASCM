import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Sample scores
scores = np.array([0.94, 0.59, 0.43, 0.47, 0.46, 0.90, 0.69, 0.92, 0.15, 0.03, 0.02, 0.08, 0.93, 0.53, 0.56, 0.83, 0.92, 0.58, 0.14, 0.60])

# Step 2: Calculate Z-Scores
mean_score = np.mean(scores)
std_dev_score = np.std(scores)
z_scores = (scores - mean_score) / std_dev_score

# Step 3: Check for Normal Distribution
plt.hist(z_scores, bins='auto', density=True, alpha=0.7, color='blue')
plt.title('Z-Scores Distribution')
plt.xlabel('Z-Scores')
plt.ylabel('Frequency')
plt.show()

# Step 4: Calculate Mean and Standard Deviation of Z-Scores
mean_z = np.mean(z_scores)
std_dev_z = np.std(z_scores)
print(f'Mean of Z-Scores: {mean_z:.2f}')
print(f'Standard Deviation of Z-Scores: {std_dev_z:.2f}')

# Step 5: Perform Shapiro-Wilk test for normality
stat, p_value = shapiro(z_scores)
print(f'Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p_value:.4f}')

# Step 6: Provide Documentation
print("\nDocumentation:")
print("Z-Scores were calculated using the formula z = (X - mean) / std_dev.")
print("The distribution of Z-Scores was checked for normality using the Shapiro-Wilk test.")

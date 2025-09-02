import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Loading the dataset with corrected file path
data = pd.read_csv(r"D:/Downloads/K-means-Clustering-for-customer-segmentations-main/K-means-Clustering-for-customer-segmentations-main/R implementations/Mall_Customers.csv")

# Convert the 'Gender' column to numerical values
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Display the first few rows
print(data.head())

# Calculate and print the correlation matrix
print(data.corr())

# Distribution of Annual Income
plt.figure(figsize=(10, 6))
sns.set(style='whitegrid')
sns.histplot(data['Annual Income (k$)'], kde=True)
plt.title('Distribution of Annual Income (k$)', fontsize=20)
plt.xlabel('Range of Annual Income (k$)')
plt.ylabel('Count')
plt.show()

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age', fontsize=20)
plt.xlabel('Range of Age')
plt.ylabel('Count')
plt.show()

# Distribution of Spending Score
plt.figure(figsize=(10, 6))
sns.histplot(data['Spending Score (1-100)'], kde=True)
plt.title('Distribution of Spending Score (1-100)', fontsize=20)
plt.xlabel('Range of Spending Score (1-100)')
plt.ylabel('Count')
plt.show()

# Gender distribution
genders = data['Gender'].value_counts()
plt.figure(figsize=(10, 4))
sns.barplot(x=genders.index, y=genders.values)
plt.xticks([0, 1], ['Male', 'Female'])
plt.show()

# Selecting relevant features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scatterplot of the input data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=X, s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()

# Elbow method for optimal K value
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1, 11, 1))
plt.ylabel("WCSS")
plt.title("Elbow Method to Determine Optimal K")
plt.show()

# Performing K-means clustering with K=5
km1 = KMeans(n_clusters=5, random_state=0)
y = km1.fit_predict(X)

# Adding the labels to the dataframe
data['label'] = y

# Scatterplot of the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='label', data=data,
                palette=['green', 'orange', 'brown', 'dodgerblue', 'red'], s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Spending Score (1-100) vs Annual Income (k$) by Cluster')
plt.legend(title='Cluster')
plt.show()

# Comparing with other clustering (Hierarchical, DBSCAN, etc.) can be done by importing those algorithms and following similar steps.

# Optional: Save the labeled data to a new CSV
data.to_csv("clustered_customers.csv", index=False)

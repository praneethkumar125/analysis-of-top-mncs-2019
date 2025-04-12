import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"C:\Users\prane\Downloads\INC 5000 Companies 2019.csv")

# Data Cleaning & Transformation
def parse_revenue(val):
    if isinstance(val, str):
        val = val.replace("$", "").strip()
        if "Billion" in val:
            return float(val.replace(" Billion", "")) * 1e9
        elif "Million" in val:
            return float(val.replace(" Million", "")) * 1e6
        elif "Thousand" in val:
            return float(val.replace(" Thousand", "")) * 1e3
    return np.nan

#Handling Missing Values
df["revenue_clean"] = df["revenue"].apply(parse_revenue)
df["workers"].fillna(df["workers"].median(), inplace=True)
df["founded"] = df["founded"].replace(0, np.nan)
df["revenue_per_worker"] = df["revenue_clean"] / df["workers"]
df["worker_growth"] = df["workers"] - df["previous_workers"]
df.dropna(subset=["revenue_clean", "founded"], inplace=True)

# Explorarity Data Analysis
df.info()
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df[["growth_%", "revenue_clean", "workers", "revenue_per_worker"]].describe())
print(" Top 10 Industries:")
print(df["industry"].value_counts().head(10))

print("\n Top 10 States:")
print(df["state"].value_counts().head(10))

avg_growth_by_year = df.groupby("founded")["growth_%"].mean()
correlation = df[["revenue_clean", "workers", "growth_%", "revenue_per_worker"]].corr()
print("\n Correlation Matrix:\n", correlation)

# Advanced Visualizations - 2x3 Dashboard Layout with Dark Theme
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("INC 5000 Companies - Dashboard Overview", fontsize=16, color="cyan")

# Chart 1: Growth % Histogram
sns.histplot(df["growth_%"], bins=50, kde=True, ax=axs[0, 0], color="deepskyblue")
axs[0, 0].set_title("Growth % Distribution")

# Chart 2: Top 10 Industries
top_industries = df["industry"].value_counts().head(10)
sns.barplot(x=top_industries.values, y=top_industries.index, ax=axs[0, 1], palette="cool")
axs[0, 1].set_title("Top 10 Industries")

# Chart 3: Avg. Growth by Founded Year
avg_growth_by_year = df.groupby("founded")["growth_%"].mean()
sns.lineplot(x=avg_growth_by_year.index, y=avg_growth_by_year.values, ax=axs[0, 2], color="lime")
axs[0, 2].set_title("Avg. Growth by Founded Year")

# Chart 4: Correlation Heatmap
correlation = df[["revenue_clean", "workers", "growth_%", "revenue_per_worker"]].corr()
sns.heatmap(correlation, annot=True, cmap="magma", ax=axs[1, 0])
axs[1, 0].set_title("Feature Correlation Matrix")

# Chart 5: Top 5 Industries Pie
top5 = df["industry"].value_counts().head(5)
axs[1, 1].pie(top5, labels=top5.index, autopct='%1.1f%%', startangle=140)
axs[1, 1].set_title("Top 5 Industries")
axs[1, 1].axis("equal")

# Chart 6: Revenue vs Workers (Log Scale)
sns.scatterplot(data=df, x="workers", y="revenue_clean", hue="growth_%",
                palette="viridis", size="growth_%", sizes=(20, 200), ax=axs[1, 2])
axs[1, 2].set_xscale("log")
axs[1, 2].set_yscale("log")
axs[1, 2].set_title("Revenue vs. Workers (log scale)")
axs[1, 2].set_xlabel("Workers")
axs[1, 2].set_ylabel("Revenue")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# STATE DISTRIBUTION - Vertical Bar Chart
top_states = df["state"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_states.index, y=top_states.values, palette="flare")
plt.title("Top 10 States by Company Count")
plt.xlabel("State")
plt.ylabel("Number of Companies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Log-Log Regression Plot (Revenue vs Workers)
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df, x="workers", y="revenue_clean", 
    scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}, 
    logx=True
)
plt.xscale('log')
plt.yscale('log')
plt.title("Revenue vs Workers (Log-Log Scale with Regression)")
plt.xlabel("Number of Workers (log scale)")
plt.ylabel("Revenue (log scale)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

#Upper-Triangle Masked Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[["growth_%", "revenue_clean", "workers", "revenue_per_worker", "worker_growth"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Feature Correlations (Masked Upper Triangle)")
plt.show()

#Pairplot for Multivariate Exploration
selected_features = ["growth_%", "revenue_clean", "workers", "revenue_per_worker"]
sns.pairplot(df[selected_features], corner=True, diag_kind="kde", plot_kws={'alpha': 0.6})
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()

# Select top 5 industries
top5_industries = df["industry"].value_counts().head(5).index
df_top5_industries = df[df["industry"].isin(top5_industries)]

# Create Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_top5_industries, x="industry", y="revenue_per_worker", palette="Set2")
plt.title("Revenue Per Worker Distribution by Top 5 Industries", fontsize=14, weight='bold')
plt.xlabel("Industry")
plt.ylabel("Revenue per Worker")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

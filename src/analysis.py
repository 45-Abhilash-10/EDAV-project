
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/sampled_100_online_course_engagement_data.csv')


# Convert 'completed' column to binary (1 = Yes, 0 = No)
df['completed'] = df['completed'].apply(lambda x: 1 if str(x).strip().lower() in ['yes','1','true','completed'] else 0)

# Q1: Compute completion rate by course_level
print("=== Q1: Completion Rate by Course Level ===")
completion_by_level = df.groupby('course_level')['completed'].mean().round(2) * 100
print(completion_by_level)

sns.barplot(x=completion_by_level.index, y=completion_by_level.values)
plt.title("Completion Rate by Course Level (%)")
plt.ylabel("Completion Rate (%)")
plt.tight_layout()
plt.savefig('visuals/q1_completion.png')
plt.close()

# Q2: Analyze hours_spent vs completion
print("=== Q2: Hours Spent vs Completion ===")
sns.boxplot(data=df, x='completed', y='hours_spent')
plt.title("Hours Spent vs Completion")
plt.tight_layout()
plt.savefig('visuals/q2_hours_vs_completion.png')
plt.close()

# Q3: Replace missing quizzes_attempted with zero
print("=== Q3: Replace Missing Quizzes Attempted ===")
before = df['quizzes_attempted'].isnull().sum()
df['quizzes_attempted'] = df['quizzes_attempted'].fillna(0)
after = df['quizzes_attempted'].isnull().sum()
print(f"Before: {before} missing values, After: {after}")

# Q4: Correlate quizzes_attempted with completion
print("=== Q4: Correlation between Quizzes Attempted and Completion ===")
correlation = df['quizzes_attempted'].corr(df['completed'])
print("Correlation:", correlation)

# Q5: Plot bar and scatter charts for engagement insights
print("=== Q5: Engagement Insights ===")
avg_quiz = df.groupby('course_level')['quizzes_attempted'].mean()
sns.barplot(x=avg_quiz.index, y=avg_quiz.values)
plt.title("Average Quizzes Attempted by Course Level")
plt.tight_layout()
plt.savefig('visuals/q5_bar.png')
plt.close()

sns.scatterplot(x='hours_spent', y='quizzes_attempted', hue='completed', data=df)
plt.title("Hours vs Quizzes Attempted (Colored by Completion)")
plt.tight_layout()
plt.savefig('visuals/q5_scatter.png')
plt.close()

print("All analysis and visualizations completed successfully!")

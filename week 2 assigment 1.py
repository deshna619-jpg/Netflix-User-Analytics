import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Q1
df = pd.read_csv("Dataset 2.csv")

print("First 5 Records:")
print(df.head())

# Q2
print("\nRows and Columns:")
print(df.shape)

# Q3
print("\nColumn Names:")
print(df.columns)

# Q4
print("\nNumerical Features:")
print(df.select_dtypes(include=['int64', 'float64']).columns)

print("\nCategorical Features:")
print(df.select_dtypes(include=['object']).columns)

# Q5
print("\nMissing Values:")
print(df.isnull().sum())

# Q6
print("\nAverage Age:")
print(df["Age"].mean())

# Q7
print("\nAverage Watch Hours Per Week:")
print(df["WatchHoursPerWeek"].mean())

# Q8
print("\nAverage Monthly Spend:")
print(df["MonthlySpend"].mean())

# Q9
print("\nUsers in Each Subscription Category:")
print(df["SubscriptionType"].value_counts())

# Q10
renewed_percent = (df["SubscriptionRenewed"] == "Yes").mean() * 100
print("\nPercentage of Renewed Users:")
print(renewed_percent)

# Q11
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Q12
X = df.drop("SubscriptionRenewed", axis=1)
y = df["SubscriptionRenewed"]

# Q13
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Q14
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Q15
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:")
print(dt_accuracy)

# Q16
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# Q17
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Q18
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:")
print(knn_accuracy)

# Q19
X_reg = df.drop("MonthlySpend", axis=1)
y_reg = df["MonthlySpend"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

# Q20
new_user = X_reg.iloc[[0]]
predicted_spend = lr.predict(new_user)

print("\nPredicted Monthly Spend for New User:")
print(predicted_spend[0])

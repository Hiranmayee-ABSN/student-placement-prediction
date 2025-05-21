# %%
import numpy as np
import pandas as pd
import matplotlib as plt
df = pd.read_csv('placement-dataset.csv')
df.info()


# %%
# Drop the index column
df.drop('Unnamed: 0', axis=1, inplace=True)
# Define features and target
X = df.drop('placement', axis=1)
y = df['placement']

# Check the data
print(X.head())
print(y.head())


# %%
from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)


# %%
# Import the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the model
model = LogisticRegression()

# Train the model on training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy on test data: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Reds')

# Show color bar
plt.colorbar(im)

# Label ticks
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(['Not Placed', 'Placed'])
ax.set_yticklabels(['Not Placed', 'Placed'])

# Rotate the tick labels and set alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()


# %%
#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate accuracy
from sklearn.metrics import accuracy_score
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# %%
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize KNN with k neighbors (e.g., k=3)
knn = KNeighborsClassifier(n_neighbors=3)

# Train model on training data
knn.fit(X_train, y_train)

# Predict on test data
y_pred_knn = knn.predict(X_test)

# Evaluate accuracy
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred_knn))


# %%
from sklearn.tree import DecisionTreeClassifier

# Initialize Decision Tree with a fixed random state for reproducibility
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Predict on test set
y_pred_dt = dt.predict(X_test)

# Evaluate accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred_dt))


# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))  # Bigger figure
plot_tree(
    dt, 
    feature_names=X.columns, 
    class_names=['Not Placed', 'Placed'], 
    filled=True, 
    proportion=True,
    impurity=False,
    fontsize=12
)
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd

importances = dt.feature_importances_
features = X.columns

plt.bar(features, importances)
plt.title("Feature Importance in Decision Tree")
plt.show()


# %%
import joblib
joblib.dump(model, 'final_model.pkl')


# %%


# %%

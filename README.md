song-popularity-predictor
Song Popularity Binary Classification using ANN with Class Weights
import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.metrics import accuracy_score, classification_report from imblearn.over_sampling import SMOTE import matplotlib.pyplot as plt import seaborn as sns import tensorflow as tf from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Dense, Dropout from sklearn.utils.class_weight import compute_class_weight import warnings warnings.filterwarnings('ignore')

Step 1: Load the dataset
df = pd.read_csv("/content/Spotify_data.csv")

Step 2: Drop non-numeric columns
df = df.select_dtypes(include=[np.number])

Step 3: Convert 'Popularity' into binary classes
y = (df['Popularity'] >= 65).astype(int) X = df.drop('Popularity', axis=1)

Step 4: Check distribution of classes
class_counts = y.value_counts() print("Class distribution before SMOTE:", class_counts.to_dict())

Step 5: Apply SMOTE if enough samples
if class_counts.min() >= 6: sm = SMOTE(random_state=42) X_sm, y_sm = sm.fit_resample(X, y) else: print("⚠ Not enough samples for SMOTE. Using original data.") X_sm, y_sm = X, y

Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

Step 7: Feature Scaling
scaler = StandardScaler() X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test)

Step 8: Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train) class_weights_dict = dict(enumerate(class_weights)) print("Class Weights:", class_weights_dict)

Step 9: Build the ANN model
model = Sequential() model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu')) model.add(Dropout(0.3)) model.add(Dense(32, activation='relu')) model.add(Dropout(0.3)) model.add(Dense(1, activation='sigmoid')) # Binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall', 'accuracy'])

Step 10: Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0, class_weight=class_weights_dict)

Step 11: Evaluate the model
y_pred_probs = model.predict(X_test_scaled) y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("ANN Binary Test Accuracy:", accuracy_score(y_test, y_pred)*100) print("\nClassification Report:\n", classification_report(y_test, y_pred))

Step 12: Plot training history
plt.figure(figsize=(12, 5)) plt.plot(history.history['accuracy'], label='Train Accuracy') plt.plot(history.history['val_accuracy'], label='Validation Accuracy') plt.title('Training History') plt.xlabel('Epochs') plt.ylabel('Accuracy') plt.legend() plt.grid(True) plt.show()

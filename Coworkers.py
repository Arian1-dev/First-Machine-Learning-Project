import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
def conversion(in_var):
    a=[]
    for i in in_var:
        a+=[i]
        
    for i in range(0, len(a)):
        a[i]=[a[i]]
        
    return a


excel_file='Data2.xlsx'

df=pd.read_excel(excel_file)


X = np.hstack([
    conversion(df['Age']),
    conversion(df['Monthly Bill ']),
    conversion(df['Contrast Length ']),
    conversion(df['Support Call '])
])
y =np.array( conversion(df['Churn ']),dtype=np.float32)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_model(input_var):
    model=Sequential([
    Dense(64,activation='relu', input_shape=(input_var,)),
    Dense(1,activation='sigmoid'),
    ])
#Compile

    model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    
    )
    return model


# Trainig without Scaling
model_no_scale = build_model(X_train.shape[1])
history_no_scale = model_no_scale.fit(
X_train, y_train,
validation_data=(X_val, y_val),
epochs=100,
batch_size=32,
verbose=0,
callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

# With Scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


print("\n===== Training WITH StandardScaler =====")
model_with_scale = build_model(X_train_scaled.shape[1])
history_with_scale = model_with_scale.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    verbose=0,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

# -------------------------------
# 6. Print number of parameters
# -------------------------------
print("\n--- Model Summary (without scaling) ---")
model_no_scale.summary()
print("\n--- Model Summary (with scaling) ---")
model_with_scale.summary()


# -------------------------------
def plot_history(hist_no, hist_with, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(hist_no.history['accuracy'], label='Train (no scale)')
    axes[0].plot(hist_no.history['val_accuracy'], label='Val (no scale)')
    axes[0].plot(hist_with.history['accuracy'], label='Train (with scale)', linestyle='--')
    axes[0].plot(hist_with.history['val_accuracy'], label='Val (with scale)', linestyle='--')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Loss
    axes[1].plot(hist_no.history['loss'], label='Train (no scale)')
    axes[1].plot(hist_no.history['val_loss'], label='Val (no scale)')
    axes[1].plot(hist_with.history['loss'], label='Train (with scale)', linestyle='--')
    axes[1].plot(hist_with.history['val_loss'], label='Val (with scale)', linestyle='--')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_history(history_no_scale, history_with_scale, "Comparison: With vs Without StandardScaler")
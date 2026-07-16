import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#data1=[]


#data2=[]

#data3=[]

#data4=[]

#data5=[]

# This Function is written to create proper style for our data before handing them to np.array
def conversion(in_var):
    a=[]
    for i in in_var:
        a+=[i]
        
    for i in range(0, len(a)):
        a[i]=[a[i]]
        
    return a
        
# Reading Excel File 
excel_file='Data.xlsx'

df=pd.read_excel(excel_file)


# Seperating each column data

#dcol1=list(df['Hour Study'])

#dcol2=list(df['Midterm'])

#dcol3=list(df['Attendence'])

#dcol4=list(df['homework '])

#dcol5=list(df['Pass '])

# Initializing data before np.array
X = np.hstack([
    conversion(df['Hour Study']),
    conversion(df['Midterm']),
    conversion(df['Attendence']),
    conversion(df['homework '])
])
y =np.array( conversion(df['Pass ']),dtype=np.float32)
#data1=np.array(conversion(dcol1),dtype=np.float32)

#data2=np.array(conversion(dcol2),dtype=np.float32)

#data3=np.array(conversion(dcol3),dtype=np.float32)

#data4=np.array(conversion(dcol4),dtype=np.float32)

#data5=np.array(conversion(dcol5),dtype=np.float32)

# Considering Three different architecture with 3 arch varaible and giving them to the dictionary
arch1=[64]


arch2 = [128, 64, 32]


arch3 = [256, 128]


configs = [
    {'arch': arch1, 'activation': 'relu', 'optimizer': Adam(learning_rate=0.001), 'name': 'Model1_ReLU_Adam'},
    {'arch': arch2, 'activation': 'relu', 'optimizer': Adam(learning_rate=0.001), 'name': 'Model2_Deep_Adam'},
    {'arch': arch3, 'activation': 'tanh', 'optimizer': Adam(learning_rate=0.001), 'name': 'Model3_Wide_Tanh_Adam'},
]
  
  
def train_evaluate_models(X, y, configs, epochs=100, test_size=0.2):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    results = []

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Training {cfg['name']}")
        print(f"Architecture: {cfg['arch']}, Activation: {cfg['activation']}")

        # Build model
        model = Sequential()
        model.add(Dense(cfg['arch'][0], input_shape=(X.shape[1],), activation=cfg['activation']))
        for units in cfg['arch'][1:]:
            model.add(Dense(units, activation=cfg['activation']))
        model.add(Dense(1, activation='sigmoid'))   # binary classification

        model.compile(optimizer=cfg['optimizer'],
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        # Parameter count
        param_count = model.count_params()

        # Store metrics
        results.append({
            'Model': cfg['name'],
            'Test Accuracy': test_acc,
            'Test Loss': test_loss,
            'Parameters': param_count
        })
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Trainable Parameters: {param_count}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(cfg['name'])
        
        #Accuracy
        ax1.plot(history.history['accuracy'], label='Train Acc')
        ax1.plot(history.history['val_accuracy'], label='Val Acc')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        #Loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        
        
        
o=train_evaluate_models(X,y,configs)
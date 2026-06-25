import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


x=np.array([[1],[2],[3],[5],[6],[7]], dtype= np.float32  )
y=np.array([[0],[0],[0],[1],[1],[1]], dtype= np.float32  )

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# Neural Networking

model=Sequential([
    Dense(4,activation='relu', input_shape=(1,)),
    Dense(1,activation='sigmoid'),
])

#Compile

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    
    )


#Train

histroy=model.fit(x_train,y_train,epochs=100)

loss_train,acc_trian=model.evaluate(x_train,y_train)

#Test
test=np.array([[4]], dtype=np.float32)
predict=model.predict(x_test)

loss_train,acc_trian=model.evaluate(x_test,predict)
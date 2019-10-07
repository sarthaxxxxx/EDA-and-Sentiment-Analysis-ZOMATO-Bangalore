from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,SpatialDropout1D
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


#Map reviews to positive and negative on the basis of the ratings provided by each user.
df=pd.read_csv('Ratings.csv')
df['Sent']=df['Rating'].apply(lambda x: 1 if int(x)>2.5 else 0)
df['Review']=df['Review'].astype('str')

#Tokenize the data and vectorize the reviews to be fed into the model
tokenizer=Tokenizer(num_words=3000,split=' ')
tokenizer.fit_on_texts(df['Review'].values)
X=pad_sequences(tokenizer.texts_to_sequences(df['Review'].values))

#Model prep
model=Sequential()
model.add(Embedding(3000,32,input_length=X.shape[1]))
model.add(LSTM(32,dropout=0.33,recurrent_dropout=0.2))
#model.add(LSTM(16,dropout=0.1,recurrent_dropout=0.1))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

#Train-Test Split
Y=pd.get_dummies(df['Sent'].astype(int)).values #used for one-hot encoding
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

history=model.fit(X_train,Y_train,verbose=2,batch_size=1600,epochs=5,validation_split=0.33)

#Choosing a certain validation size
valid=2000 #Validating on 2000 rows of the test set
X_validate=X_train[-valid:]
Y_validate=Y_train[-valid:]
score,acc=model.evaluate(X_validate,Y_validate,verbose=2,batch_size=1600)
score_test,acc_test=model.evaluate(X_test,Y_test,verbose=2,batch_size=1600)
print("Score on validation set: %0.2f"%(score))
print("Accuracy of validation set: %0.2f"%(acc))
print("Score on test set: %0.2f"%(score_test))
print("Accuracy of test set: %0.2f"%(acc_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
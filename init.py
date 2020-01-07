import model
from tensorflow.python.keras.optimizers import SGD
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.models import model_from_json
import tensorflow as tf
import datetime

data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

dataset = data[0].reshape((data[0].shape[0], 28, 28, 1))
labels = np.asanyarray(data[1])

le = LabelBinarizer()
le.fit(labels)
labels = le.transform(labels)

X_train, X_test, Y_train, Y_test = train_test_split(
    dataset/255.0, labels, test_size=0.25)

model = model.LeNet.build((28, 28, 1))

sgd = SGD(nesterov=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["accuracy"])

H = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              batch_size=128, epochs=10, verbose=1, callbacks=[tensorboard_callback])

predictions = model.predict(X_test, batch_size=128)
print(classification_report(Y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

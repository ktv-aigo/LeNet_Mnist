from tensorflow.keras import models, layers

class LeNet:
    def build(shape):
        model = models.Sequential()
        model.add(layers.Conv2D(20, (5,5), activation='relu',padding = 'same', input_shape=shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(50, (5, 5), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

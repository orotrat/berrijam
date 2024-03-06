import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import legacy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

class ImageNormaliser:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def get_training_generator(self, train_dir, batch_size):
        """Creates and returns a training data generator."""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode='binary'
        )

        return train_generator

    def get_test_generator(self, validation_dir, batch_size):
        """Creates and returns a validation data generator."""
        test_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size,
            class_mode='binary'
        )

        return validation_generator

def build_model(img_width, img_height):
    # Load the MobileNetV2 model, excluding its top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(img_width, img_height, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # Add a logistic layer -- we have 2 classes (cats and dogs)
    predictions = Dense(1, activation='sigmoid')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    # i.e., freeze all convolutional MobileNetV2 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

def evaluate_model(model, test_generator):
    # Ensure the test_generator does not use shuffling to yield consistent predictions
    test_generator.shuffle = False
    test_generator.batch_size = 1

    # Predict on the test data
    Y_pred = model.predict(test_generator, steps=len(test_generator))
    y_pred = np.where(Y_pred > 0.5, 1, 0).reshape(-1)  # Convert probabilities to binary predictions

    # True labels
    y_true = test_generator.classes

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

def main(build_cat_dog=False, build_fruits=False):
    img_width, img_height = 224, 224
    batch_size = 32

    train_dir = "data/train/"
    test_dir = "data/test/"

    image_normaliser = ImageNormaliser(img_width, img_height)

    train_generator = image_normaliser.get_training_generator(train_dir, batch_size)
    test_generator = image_normaliser.get_test_generator(test_dir, batch_size)

    print("Testing cats vs dogs classifier...")

    if build_cat_dog:
        model = build_model(img_width, img_height)
        # model.compile(optimizer=Adam(learning_rate=0.0001),
        #               loss='binary_crossentropy',
        #               metrics=['accuracy'])
        optimizer = legacy.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        

        model.fit(train_generator, epochs=10,
                validation_data=test_generator)

        # Save the trained model
        model.save('cats_vs_dogs_classifier.keras')
    else:
        model = load_model('cats_vs_dogs_classifier.keras')

    evaluate_model(model, test_generator)

    fruits_train_dir = "fruits/train/"
    fruits_test_dir = "fruits/test/"

    fruits_train_generator = image_normaliser.get_training_generator(fruits_train_dir, batch_size)
    fruits_test_generator = image_normaliser.get_test_generator(fruits_test_dir, batch_size)

    print("Testing fruits classifier...")

    if build_fruits:
        model = build_model(img_width, img_height)
        # model.compile(optimizer=Adam(learning_rate=0.0001),
        #               loss='binary_crossentropy',
        #               metrics=['accuracy'])
        optimizer = legacy.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        

        model.fit(fruits_train_generator, epochs=10,
                validation_data=fruits_test_generator)

        # Save the trained model
        model.save('fruits_classifier.keras')
    else:
        model = load_model('fruits_classifier.keras')

    evaluate_model(model, fruits_test_generator)

if __name__ == "__main__":
    main()
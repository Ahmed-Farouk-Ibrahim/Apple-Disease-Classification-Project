from collections import Counter
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop 
from sklearn.utils import class_weight
import numpy as np

class TrainModel:
    """
    TrainModel class is responsible for loading a prebuilt model,
    training it with specified data, and saving the trained model.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the TrainModel with the given configuration.

        Args:
            config (TrainingConfig): Configuration object for training the model.
        """
        self.config = config

    def data_generator(self):
        """
        Load and preprocess the training and validation data.
        """
        # Define data augmentation for training data
        train_datagen = ImageDataGenerator(
            rescale=1./255,               # Rescale pixel values to [0, 1]
            rotation_range=10,            # Rotate images randomly by up to 10 degrees
            width_shift_range=0.1,        # Shift images horizontally by up to 10% of the width
            zoom_range=0.1,               # Zoom in or out by up to 10%
            horizontal_flip=True          # Flip images horizontally (left to right)
        )
        
        self.train_generator = train_datagen.flow_from_directory(
            f'{self.config.training_data}/train',
            target_size=self.config.all_params.IMAGE_SIZE[:-1],
            batch_size=self.config.all_params.BATCH_SIZE,
            class_mode='sparse',
            shuffle=True
        )

        # Define data preprocessing for validation data
        validation_datagen = ImageDataGenerator(rescale=1./255)
            
        self.validation_generator = validation_datagen.flow_from_directory(
            f'{self.config.training_data}/val',
            target_size=self.config.all_params.IMAGE_SIZE[:-1],
            batch_size=self.config.all_params.BATCH_SIZE,
            class_mode='sparse',
            shuffle=False
        )
    
    def calculate_class_weights(self):
        """
        Calculate class weights to handle class imbalance.
        """
        y_train = self.train_generator.classes
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        self.class_weights = dict(enumerate(class_weights))

    def get_built_model(self):
        """
        Load the built model from the specified path.
        """
        self.model = tf.keras.models.load_model(self.config.built_model_path)

    def train_model(self):
        """
        Train the model using the loaded data and specified parameters.
        Save the trained model to the specified path.
        """

        # Compile the model
        optimizer = RMSprop(
            learning_rate=self.config.all_params.LEARNING_RATE,
            rho=self.config.all_params.RHO,
            epsilon=self.config.all_params.EPSILON
        )
        self.model.compile(
            optimizer=optimizer, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"]
        )
    
        # Define the learning rate annealer
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            patience=self.config.all_params.PATIENCE,
            verbose=self.config.all_params.VERPOSE,
            factor=self.config.all_params.FACTOR,
            min_lr=self.config.all_params.MIN_LR
        )

        # Calculate steps per epoch and validation steps
        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.validation_generator.samples // self.validation_generator.batch_size

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.all_params.EPOCHS,
            validation_data=self.validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[reduce_lr],
            class_weight=self.class_weights
        )

        # Save the trained model architecture to JSON
        model_json = self.model.to_json()
        
        with open(self.config.trained_model_path.with_suffix(".json"), "w") as json_file:
            json_file.write(model_json)

        # Save model weights separately using the save_weights method. 
        # This ensures compatibility across different versions of TensorFlow/Keras.
        self.model.save_weights(self.config.trained_model_path.with_suffix(".h5"))
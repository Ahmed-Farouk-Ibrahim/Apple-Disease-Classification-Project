from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareModelConfig
import tensorflow as tf
from keras import models, layers
from keras.optimizers import RMSprop 

# Class for preparing and building the model
class PrepareModel:
    """
    PrepareModel class is responsible for creating a custom model,
    compiling it, and saving it to specified paths.
    """

 

    def __init__(self, config: PrepareModelConfig):
        """
        Initialize the PrepareModel with the given configuration.

        Args:
            config (PrepareModelConfig): Configuration object for preparing the model.
        """
        self.config = config

    
    @staticmethod
    def prepare_model(input_shape, n_classes, learning_rate, rho, epsilon):
        """
        Prepare the model by defining its architecture and compiling it.

        Args:
            input_shape (tuple): Shape of the input images, derived from params_image_size in the config.
            n_classes (int): Number of output classes.
            learning_rate (float): Learning rate for the optimizer.
            rho (float): Decay rate for RMSprop optimizer.
            epsilon (float): Small value to prevent division by zero in RMSprop optimizer.

        Returns:
            tf.keras.Model: The compiled model.
        """

        # Define the model architecture
        model = models.Sequential([
                layers.InputLayer(input_shape=input_shape),
                layers.Conv2D(filters = 128, kernel_size = (3,3), kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape = input_shape),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPool2D((2,2)),

                layers.Conv2D(128, (3,3), activation = 'relu'),
                layers.MaxPool2D((2,2)),

                layers.Conv2D(64, (3,3) , kernel_regularizer=tf.keras.regularizers.l2(0.01),),
                layers.BatchNormalization(),
                layers.Activation('relu'),

                layers.MaxPool2D((2,2)),
                layers.Conv2D(64, (3,3), activation = 'relu'),
                layers.MaxPool2D((2,2)),

                layers.Conv2D(64, (3,3) , kernel_regularizer=tf.keras.regularizers.l2(0.01),),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPool2D((2,2)),

                layers.Conv2D(32, (3,3), activation = 'relu'),
                layers.MaxPool2D((2,2)),

                layers.Flatten(),
                layers.Dense(32, activation = 'relu'),
                layers.Dense(n_classes, activation = 'softmax')
                ])
        # Define the optimizer :
        optimizer = RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon)
        # Compile the model:
        model.compile(
            optimizer=optimizer, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"])

        
        # Print the model summary
        model.summary()
        return model

    def build_model(self):
        """
        Build the model by adding custom layers and compile it.
        Save the updated model to the specified path.
        """

        # Prepare the full model with custom layers and specified parameters
        self.full_model = self.prepare_model(
            input_shape = self.config.params_image_size, 
            n_classes = self.config.params_n_classes, 
            learning_rate = self.config.params_learning_rate, 
            rho = self.config.params_rho, 
            epsilon = self.config.params_epsilon,
        )

        # Save the updated full model to the specified path
        self.save_model(path=self.config.built_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

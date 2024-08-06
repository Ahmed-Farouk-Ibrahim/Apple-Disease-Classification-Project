import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop 


from cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def test_generator(self):
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            horizontal_flip=True
            )

        self.test_generator = test_datagen.flow_from_directory(
            f'{self.config.training_data}/test',
            target_size=self.config.all_params.IMAGE_SIZE[:-1],
            batch_size=self.config.all_params.BATCH_SIZE,
            class_mode="sparse",
            shuffle=False,
            )
        
    def load_model(self):
        try:
            # Load model architecture
            with open(self.config.trained_model_json_path, 'r') as json_file:
                model_json = json_file.read()
            self.model = tf.keras.models.model_from_json(model_json)
            
            # Load model weights
            self.model.load_weights(self.config.trained_model_weights_path)

            ''' 
            When you load the model using tf.keras.models.load_model(path), the training configuration is loaded, 
            but sometimes it doesn't include the optimizer settings. Thus, you need to recompile it.

            Evaluation: Before evaluating the loaded model in model_evaluation.py, 
            recompiling ensures that the model has the necessary settings for loss and metrics to perform evaluation.
            '''

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

        except Exception as e:
            raise RuntimeError("Error loading model architecture or weights") from e


    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def evaluation(self):
        self.score = self.model.evaluate(self.test_generator)
        self.save_score()
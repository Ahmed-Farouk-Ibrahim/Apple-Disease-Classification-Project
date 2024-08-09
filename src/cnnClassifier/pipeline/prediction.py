import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        """
        Initialize the PredictionPipeline with the filename of the image to be predicted.
        """
        self.filename = filename

    def predict(self):
        """
        Predict the class of the image using the pre-trained model.

        Returns: str: A string containing the prediction result.
        """
        # Load the model architecture from JSON
        with open(os.path.join("model", "model.json"), "r") as json_file:
            model_json = json_file.read()
            model = model_from_json(model_json)

        # Load the model weights
        model.load_weights(os.path.join("model", "model.h5"))

        # Load and preprocess the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict the class of the image using the model
        result = np.argmax(model.predict(test_image), axis=1)

        # Map the prediction result to the corresponding class label
        class_labels = {
            0: 'Apple___Apple_scab',
            1: 'Apple___Black_rot',
            2: 'Apple___Cedar_apple_rust',
            3: 'Apple___healthy'
        }

        prediction = class_labels.get(result[0], "Unknown")

        # Mapping from model output to desired output
        label_mapping = {
            "Apple___Apple_scab": "Apple Scab",
            "Apple___Black_rot": "Black Rot",
            "Apple___Cedar_apple_rust": "Apple Rust",
            "Apple___healthy": "Healthy"
        }

        # Get the human-readable label
        human_readable_label = label_mapping.get(prediction, "Unknown")

        # Return the human-readable label
        return [human_readable_label]

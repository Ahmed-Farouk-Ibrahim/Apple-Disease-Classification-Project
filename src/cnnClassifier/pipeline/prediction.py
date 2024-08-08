import numpy as np
from keras.models import load_model
from keras.preprocessing import image
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

        Returns:   list: A list containing a dictionary with the prediction result.
        """
        # Load the trained model from the specified path
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load and preprocess the image
        imagename = self.filename
        # Resize the image to 256*256 pixels
        test_image = image.load_img(imagename, target_size=(256, 256))
        # Convert the image to an array  
        test_image = image.img_to_array(test_image)  
        # Add an extra dimension to match the model's input shape
        test_image = np.expand_dims(test_image, axis=0)  

        # Predict the class of the image using the model. Get the class with the highest probability
        result = np.argmax(model.predict(test_image), axis=1)  

        # Map the prediction result to the corresponding class label
        class_labels = {
            0: 'Apple___Apple_scab',
            1: 'Apple___Black_rot',
            2: 'Apple___Cedar_apple_rust',
            3: 'Apple___healthy'
        }

        prediction = class_labels.get(result[0], "Unknown")
        return [{"image": prediction}]
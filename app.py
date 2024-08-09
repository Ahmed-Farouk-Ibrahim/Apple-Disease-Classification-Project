from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables for language settings
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) to handle requests from different domains
CORS(app)

# ClientApp class to manage image filename and prediction pipeline
class ClientApp:
    def __init__(self):
        # Define the filename for the input image
        self.filename = "inputImage.jpg"
        
        # Initialize the prediction pipeline with the specified filename
        self.classifier = PredictionPipeline(self.filename)

# Define the home route that serves the main webpage
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')  # Render the home page template

# Define the route to trigger the model training process
@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    # Run the training script using the system shell
    #os.system("python main.py")  
    # Optionally, you can run a DVC (Data Version Control) pipeline
    os.system("dvc repro")
    return "Training done successfully!"  # Return a success message after training

# Define the route to handle image prediction requests
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Get the image data from the request
    image = request.json['image']
    
    # Decode the image and save it with the specified filename
    decodeImage(image, clApp.filename)
    
    # Use the classifier to predict the class of the image
    result = clApp.classifier.predict()
    
    # Return the prediction result as a JSON response
    return jsonify(result)

# Run the Flask application on the specified host and port (suitable for deployment on AWS)
if __name__ == "__main__":
    # Initialize the client application
    clApp = ClientApp()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=8080)  # Running on port 8080, accessible from any IP

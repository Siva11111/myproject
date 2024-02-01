from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image  # Import Image module from Pillow

app = Flask(__name__)
loaded_model = load_model('ashwin.h5')
# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    image_file = request.files['file']
    
    # Use PIL to open the image
    img = Image.open(image_file)
    img = img.resize((224, 224))  # Resize the image to the target size

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to values between 0 and 1

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Get the class label (assuming binary classification)
    class_label = "Class 1" if predictions[0][0] > 0.5 else "Class 0"

    # Print the predictions
    print("Predictions:", predictions)
    print("Class Label:", class_label)
    return render_template('upload.html',prediction=class_label) 
    
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("Home.html")

@app.route('/aboutus', methods=['GET', 'POST'])
def about():
    return render_template("AboutUs.html")

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow import keras
app = Flask(__name__)

# Globals
IMG_SIZE = (128,128)
CATEGORIES = {'buildings': 0,
              'forest': 1,
              'glacier': 2,
              'mountain': 3,
              'sea': 4,
              'street': 5 }

@app.route('/', methods=['GET', 'POST'])
def homepage():
    """
    Display the home page and display the submitted image
    and corresponding prediction after file upload
    """

    if request.method == 'GET':
        print("GET")
        return render_template('index.html.twig')
    else:
        print("POST")
        # Upload image
        image_path = upload_file()

        # Preprocess image
        image = preprocess_image(image_path)
        img_name = image_path.split('/')[-1]

        # Make prediction
        image = image.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
        print(image.shape)
        predictedLabel, confidence = predict(image)
        confidence = "{:.2f}".format(round(confidence * 100, 2))

        # Display result
        return render_template('index.html.twig',
                               predictedLabel = predictedLabel,
                               confidence = confidence, # two decimal places
                                image_name = img_name)


@app.route('/about', methods=['GET'])
def about_page():
    """
    Display about page
    """
    return render_template('about.html.twig')


def upload_file():
    """
    Handle file upload after form submission
    """

    # Get the file from post request
    file = request.files['imageToPredict']

    # Save the file to static/images/uploads
    file_path = os.path.join('static/images/uploads/', secure_filename(file.filename))

    file.save(file_path)

    return file_path

def preprocess_image(image_path):
    """
    Preprocess image before feeding the model for prediction:
    - Load in grayscale
    - Resize image
    - Cast to array and
    - Scale
    :param image_path: path to the image file (uploaded file)
    :return: preprocessed image ready to be fed to the model
    """

    # Load in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize image
    img = cv2.resize(img, IMG_SIZE)

    # plt.imshow(img, cmap='gray')
    # img_name = image_path.split('/')[-1]
    # print(img_name)
    # img_name = re.sub('.jpg', 'gray.jpg', img_name)
    # plt.imsave("static/images/uploads/grayscale" + img_name, img)

    # Cast array
    img = np.array(img)

    # Scale
    img = img/255

    # Return processsed image
    return img

def predict(image):
    """
    Make prediction for the image in parameter
    :param image: Image uploaded, predict class of this image
    :return: predicted label and confidence
    """

    # Load model
    cnn_model = keras.models.load_model('models/intel_image_classifier.h5')

    # print(cnn_model.summary())
    # tf.keras.utils.plot_model(cnn_model, to_file='static/images/model.png', show_shapes=True)

    # Predict image
    pred = cnn_model.predict(image)

    # Get highest probability index
    pred_idx = np.argmax(pred)
    print("pred_idx")
    print(pred_idx)

    # Get corresponding label string
    label = list(CATEGORIES.keys())[list(CATEGORIES.values()).index(pred_idx)]

    # Return label and prediction confidence (probability)
    return label, pred[0][pred_idx]

if __name__ == '__main__':
    app.run(None, None, True)

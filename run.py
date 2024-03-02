from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from os.path import join
from time import time
from hashlib import md5
from img2Sketch import Sketch
import tensorflow as tf
import numpy as np


UPLOAD_FOLDER = "./static/media/"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(file_object):
    img = tf.io.decode_image(file_object.read(), channels=3)
    img = tf.image.resize(img, [256, 256])
    img = (tf.cast(img, tf.float32) / 127.5) - 1
    img = tf.expand_dims(img, axis=0)  # Add batch dimension for prediction
    return img

@app.route('/')

@app.route('/uploader',methods=["GET","POST"])
def upload_file():
    if request.method == 'POST':
        file_object = request.files['file']
        filename = secure_filename(md5(str(time()).encode()).hexdigest() + '.png')
        file_path = join(app.config["UPLOAD_FOLDER"], filename)

        # Process the uploaded image
        input_image = process_image(file_object)

        # Make prediction using the loaded_styled_generator
        loaded_styled_generator = tf.keras.models.load_model('D:/Major_Project/saved_model/styled_generator')
        prediction = loaded_styled_generator(input_image, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

        # Save the processed image
        tf.keras.preprocessing.image.save_img(file_path, prediction)

        return render_template('index.html', file_url=file_path)
    else:
        return render_template('index.html')
    



if __name__ == '__main__':
    app.run(debug = True)  
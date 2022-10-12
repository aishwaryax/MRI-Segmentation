# main.py

from flask import Blueprint, render_template, request, send_from_directory
import os, sys
from PIL import Image
import errno
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf



ALLOWED_EXTENSIONS = {'tif'}

from os.path import join, dirname, realpath
UPLOADS_PATH = dirname(realpath(__file__)) + '/uploads/'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



main = Blueprint('main', __name__)


@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict')
def form():
    return render_template('form.html')

@main.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'upload' not in request.files:
            flash('No file part')
            return render_template("form.html")
        file = request.files['upload']
        from datetime import datetime
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            UPLOADS_PATH = dirname(realpath(__file__)) + '/uploads/' + current_user.name + "/" + str(timestamp)
            filename = UPLOADS_PATH + "/mri/" + filename
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            file.save(filename)
            mriTiff = filename
            predict_path = UPLOADS_PATH + "/"
            target_img_shape_1 = 256
            target_img_shape_2 = 256
            from keras.preprocessing.image import ImageDataGenerator
            predict_datagen = ImageDataGenerator(rescale = 1./255)
            predict_generator = predict_datagen.flow_from_directory(predict_path, target_size = (target_img_shape_1, target_img_shape_2)) 
            with open(dirname(realpath(__file__)) + '/classifier-resnet-model.json', 'r') as json_file:
                json_savedModel= json_file.read()
            
            model = tf.keras.models.model_from_json(json_savedModel)
            path = dirname(realpath(__file__)) + "/classifier-resnet-weights.hdf5"
            model.load_weights(path)
            model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])
            predict = model.predict(predict_generator)
            if (predict[0][0]> predict[0][1]):
                string_predict = "Tumor Absent"
            else: 
                string_predict = "Tumor Present"
            with open(dirname(realpath(__file__)) + '/ResUNet-MRI.json', 'r') as json_file:
                json_savedModel= json_file.read()
            model = tf.keras.models.model_from_json(json_savedModel)
            path = dirname(realpath(__file__)) + "/ResUNet-weights.hdf5"
            model.load_weights(path)
            model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])
            pred = model.predict(predict_generator)
            import numpy as np
            img = np.squeeze(pred)
            import matplotlib.image

            #mri
            im = Image.open(mriTiff)
            out = im.convert("RGB")
            out.save(dirname(realpath(__file__)) + '/uploads/' + current_user.name + "_" + str(timestamp) + '_mri.png', "JPEG", quality=90)

            matplotlib.image.imsave(dirname(realpath(__file__)) + '/uploads/' + current_user.name + "_" + str(timestamp) + '_mask.png', img)
            mask_filename = 'http://127.0.0.1:5000/uploads/' + current_user.name + "_" + str(timestamp) + '_mask.png'
            mri_filename = 'http://127.0.0.1:5000/uploads/' + current_user.name + "_" + str(timestamp) + '_mri.png'
            return render_template("predict.html", predict = str(string_predict), mask_filename = mask_filename, mri_filename = mri_filename)


@main.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOADS_PATH, filename)

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

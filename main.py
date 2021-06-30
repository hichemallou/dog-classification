import os
from flask import Flask, request
import numpy as np
import tensorflow as tf

from catNames  import  cat

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



UPLOAD_FOLDER = 'D:\\'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
image_gen_train = ImageDataGenerator(rescale=1. / 255,

                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     )
model = keras.models.load_model('C:\\Users\\33652\\PycharmProjects\\dogClassification\\InceptionV3_120')
IMG_SHAPE = 224
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)

        file1.save(path)

        img = load_img(path, target_size=(IMG_SHAPE, IMG_SHAPE))

        data = img_to_array(img)

        samples = expand_dims(data, 0)

        it = image_gen_train.flow(samples, batch_size=1)
        res = model.predict(it)
        res.max(), np.argmax(res, axis=-1), cat[np.argmax(res)]
        print(res.max(), np.argmax(res, axis=-1), cat[np.argmax(res)])

        return  cat[np.argmax(res)]

        return 'ok'
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    '''

if __name__ == '__main__':
    app.run()
import json
import time
from io import BytesIO
import datetime
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request


application = Flask(__name__)

model = None

def load_model():     
    global model
    if (model == None):
        print('Attempting to load model')
        model = tf.keras.models.load_model('/app/model.h5')
        model.summary()
        print('Done!')    

   

@application.route("/", methods=['GET', 'POST'])
def run():
    if (request.method == 'GET'):
        return "Healthy"
    else:
        load_model()

        prev_time = time.time()

        post = request.get_json()
        print(post)
        img_path = post['image']

        current_time = time.time()

        tensor = process_image(img_path, 160)
        t = tf.reshape(tensor, [-1, 160, 160, 3])
        o = model.predict(t, steps=1)  # [0][0]
        print(o)
        o = o[0][0]
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        payload = {
            'time': inference_time.total_seconds(),
            'prediction': 'burrito' if o > 0.5 else 'tacos',
            'scores': str(o)
        }

        print('Input ({}), Prediction ({})'.format(post['image'], payload))

        return payload


def process_image(path, image_size):
    # Extract image (from web or path)
    if path.startswith('http'):
        response = requests.get(path)
        img = np.array(Image.open(BytesIO(response.content)))
    else:
        img = np.array(Image.open(path))

    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    # tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, [image_size, image_size]) / 255
    return img_final


def info(msg, char="#", width=75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1 * width) + 5, msg) + char)
    print(char * width)


if __name__ == "__main__":
    application.run(host='0.0.0.0')

# if __name__ == "__main__":
#     images = {
#         'tacos': 'https://c1.staticflickr.com/5/4022/4401140214_f489c708f0_b.jpg',  # noqa: E501
#         'burrito': 'https://www.exploreveg.org/files/2015/05/sofritas-burrito.jpeg'  # noqa: E501
#     }

#     init()

#     for k, v in images.items():
#         print('{} => {}'.format(k, v))

#     info('Taco Test')
#     taco = json.dumps({'image': images['tacos']})
#     print(taco)
#     run(taco)

#     info('Burrito Test')
#     burrito = json.dumps({'image': images['burrito']})
#     print(burrito)
#     run(burrito)

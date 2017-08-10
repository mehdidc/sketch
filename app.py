import sys

import click

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.filters import threshold_otsu
from flask import Flask, request, redirect, render_template, url_for, Response, jsonify
import base64
import numpy as np

from machinedesign.autoencoder.interface import load
from machinedesign.transformers import transform_one, inverse_transform_one

model = load('models/mnist')

DEBUG = True
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@click.command()
@click.option('--host', default='0.0.0.0', required=False)
@click.option('--port', default=20004, required=False)
def serve(host, port):
    app.run(host=host, port=port, debug=DEBUG)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('sketch.html')

@app.route('/fill')
def fill():
    img = request.args.get('img', '', type=str)
    img = process_img(img, _fill)
    return jsonify(img=img)

@app.route('/create')
def create():
    img = request.args.get('img', '', type=str)
    img = process_img(img, _create)
    return jsonify(img=img)


def _create(data):
    size = 10
    y = np.random.randint(0, data.shape[2] - size)
    x = np.random.randint(0, data.shape[3] - size)
    shape = data[:, :, y:y+size, x:x+size].shape
    data[:, :, y:y+size, x:x+size] = np.random.uniform(size=shape) 
    for i in range(1):
        data = model.predict(data)
        data = data.astype(np.float32)
    return data

def _fill(data):
    for i in range(1):
        data = transform_one(data, model.transformers)
        data = model.predict(data)
        data = inverse_transform_one(data, model.transformers)
        data = data.astype(np.float32)
    return data

def process_img(img, func):
    header, content = img.split(',', 2)
    with open('img.png', 'wb') as fd:
        d = base64.b64decode(content)
        fd.write(d)
    img = imread('img.png')
    img = resize(img, (28, 28), preserve_range=True)
    data = img[np.newaxis, :, :, :].astype(np.float32)
    data = data.transpose((0, 3, 1, 2))
    if model.layers[0].output_shape[1] == 1:
        data = data[:, 3:4, :, :]
    else:
        data = data[:, 0:3, :, :]
    data /= 255.
    data = func(data)
    data = data.transpose((0, 2, 3, 1))

    img = data[0]
    if model.layers[0].output_shape[1] == 1:
        img = img[:, :, 0]
        img = img[:, :, None] * np.ones((1, 1, 4))
        img[:, :, 0:3]=0
        img = img * 255.
        img = img.astype(np.uint8)
    else:
        pass
    #img = resize(img, (256, 256))
    imsave('img.png', img)
    data = open('img.png', 'rb').read()
    content = base64.b64encode(data)
    content = content.decode("utf-8")
    img  = "{header},{content}".format(header=header, content=content)
    return img

if __name__ == '__main__':
    serve()

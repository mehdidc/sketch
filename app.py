import sys

import click

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.filter import threshold_otsu
from flask import Flask, request, redirect, render_template, url_for, Response, jsonify
import base64

import numpy as np

sys.path.append('/root/work/code/feature_generation')
from tools.common import load_model_simple

model_path = '/root/work/code/feature_generation/jobs/results/1b5f929796b52352a009ab37f602bfbf/model.pkl'
model = load_model_simple(model_path, w=28, h=28)

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
    img = process_img(img, fill_)
    return jsonify(img=img)

@app.route('/create')
def create():
    img = request.args.get('img', '', type=str)
    img = process_img(img, create_)
    return jsonify(img=img)


def create_(data):
    size = 10
    y = np.random.randint(0, data.shape[2] - size)
    x = np.random.randint(0, data.shape[3] - size)
    shape = data[:, :, y:y+size, x:x+size].shape
    data[:, :, y:y+size, x:x+size] = np.random.uniform(size=shape) 
    #data[:] = np.random.uniform(size=data.shape)
    for i in range(1):
        data = model.reconstruct(data)
        thresh = moving_thresh(data, 0.17)
        data = data > thresh
        data = data.astype(np.float32)
    return data

def fill_(data):
    for i in range(1):
        data = model.reconstruct(data)
        thresh = moving_thresh(data, 0.17)
        data = data > thresh
        data = data.astype(np.float32)
    return data

def moving_thresh(s, whitepx_ratio):
    vals = s.flatten()
    vals = vals[np.argsort(vals)]
    thresh_ = vals[-int(whitepx_ratio * len(vals)) - 1]
    return thresh_

def process_img(img, func):
    header, content = img.split(',', 2)
    with open('img.png', 'wb') as fd:
        d = base64.b64decode(content)
        fd.write(d)
    img = imread('img.png')
    pad = 2
    #img = np.pad(img, (pad, pad), 'constant', constant_values=0)

    img = img[:, :, 3]
    #img = resize(img, (28, 28), preserve_range=True)
    try:
        img = img > threshold_otsu(img)
    except Exception:
        pass

    data = img[None, :, :, None].astype(np.float32)
    data = data.transpose((0, 3, 1, 2))
    data = func(data)
    data = data.transpose((0, 2, 3, 1))
    img = data[0]
    img = img[:, :, 0]
    #img = img > threshold_otsu(img)
    #img = resize(img, (400, 400), preserve_range=True)
    #img = img[pad:-pad, pad:-pad]
    img = img[:, :, None] * np.ones((1, 1, 4))
    img[:, :, 0:3]=0
    img = img * 255.
    img = img.astype(np.uint8)
    imsave('img.png', img)
    content = base64.b64encode(open('img.png').read())
    img  = "{header},{content}".format(header=header, content=content)
    return img

if __name__ == '__main__':
    serve()

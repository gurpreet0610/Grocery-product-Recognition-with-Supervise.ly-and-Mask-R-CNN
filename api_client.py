import requests
import os
image_path='/home/gurpreet/Code/github/CocoColaFantaSrite/main.jpg'
headers = {
    'X-API-KEY': os.environ['SUPERVISELY_API_TOKEN'],
}

files = (
    ('id', (None, '19449')),
    ('data', (None, '{}')),
    ('image', ('mypicture.jpg;type=image/jpeg', open(image_path, 'rb')))
)

response = requests.post('https://app.supervise.ly/public/api/v3/models.infer', headers=headers, files=files)
with open('response.json', 'wb') as outf:
    outf.write(response.content)


# Supervisely SDK
import supervisely_lib as sly
import json 


def load_metadata():

    # Opening JSON file 
    with open('meta.json', 'r') as openfile: 
    
        # Reading from json file 
        json_object = json.load(openfile) 
    meta = sly.ProjectMeta.from_json(json_object)

    return meta


import numpy as np

ann = sly.Annotation.load_json_file("response.json",load_metadata())

import matplotlib.pyplot as plt

def draw_labeled_image(img, ann, draw_fill=False):
    canvas_draw_contour = img.copy()
    if ann is not None:
        if draw_fill is True:
            ann.draw(canvas_draw_contour)
        else:
            ann.draw_contour(canvas_draw_contour, thickness=7)
    fig = plt.figure(figsize=(30, 30))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    if ann is not None:
        fig.add_subplot(1, 2, 2)
        plt.imshow(canvas_draw_contour)    
    plt.show() 

draw_labeled_image(sly.image.read(image_path), ann, draw_fill=True)
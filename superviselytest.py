import supervisely_lib as sly
import json
img = sly.image.read('test_image.jpg')

address ='https://app.supervise.ly/'
token = "ntfCVxQIydbR6JCDPiii5mX1cIFdgA9QjcVBdudlx3Z1Y5PTUXFbFfkqkCu6y11DIgDQWPseRVIgHtmUs0mV4x5FY8I9Ch93okonzRmHAMUNYdEFe4nAaZBEMUk2yTss"
api = sly.Api(address, token)


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
def load_metadata():
    
    # Opening JSON file 
    with open('meta.json', 'r') as openfile: 
    
        # Reading from json file 
        json_object = json.load(openfile) 
    meta = sly.ProjectMeta.from_json(json_object)

    return meta

ann_json = api.model.inference(19449, img)
ann_seg = sly.Annotation.from_json(ann_json,load_metadata() )

# Render the inference results.
draw_labeled_image(img, ann_seg, draw_fill=True)
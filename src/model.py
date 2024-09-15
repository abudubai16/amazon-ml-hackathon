import torch
from PIL import Image
from src.cleaning import  clean_text

from ikomia.dataprocess.workflow import Workflow
from PIL import Image
from IPython.display import display

def text_from_image_ikomia(image_url):
    # Init your workflow
    wf = Workflow()

    # Add text detection algorithm
    text_det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

    # Add text recognition algorithm
    text_rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)

    wf.run_on(url=image_url)

    img_output = text_rec.get_output(0)
    recognition_output = text_rec.get_output(1)

    image_data = img_output.get_image_with_mask_and_graphics(recognition_output)
    display(Image.fromarray(image_data))
    
    text_fields = recognition_output.get_text_fields()
    texts = ''
    for text_field in text_fields:
        texts += text_field.text + ' '

    return texts

def final_model(image_url,entity_name):
    text = clean_text(text_from_image_ikomia(image_url), entity_name=entity_name)
    print(text)
    return text

if __name__ == '__main__':
    
    a = final_model('https://m.media-amazon.com/images/I/110EibNyclL.jpg', 'width')
    print(a)
    

    pass
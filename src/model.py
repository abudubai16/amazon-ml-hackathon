import torch
from PIL import Image


def text_from_image_ikomia(image_url):
    from ikomia.dataprocess.workflow import Workflow
    from PIL import Image
    from IPython.display import display

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

    texts = [text_field.text for text_field in text_fields]

    return texts


if __name__ == '__main__':
    
    a = text_from_image_ikomia('https://m.media-amazon.com/images/I/110EibNyclL.jpg')
    print(a)
    

    pass
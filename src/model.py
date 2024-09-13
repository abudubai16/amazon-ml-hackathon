from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

PATH = r'C:\Users\Abhyuday Chauhan\PycharmProjects\student_resource 3\images\41-NCxNuBxL.jpg'

def extract_text_from_image(image_path):
    image = Image.open(image_path)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if __name__ == '__main__':
    text = extract_text_from_image(PATH)
    print(text)
    pass
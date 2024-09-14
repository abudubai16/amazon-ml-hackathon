import torch
from PIL import Image



def text_from_image_easyocr(image_path):
    import easyocr
    img = Image.open(image_path)

    reader = easyocr.Reader(['en'])  # specify language
    results = reader.readtext(img)

    words = []
    for result in results:
        words.append(result[1])

    return words

def text_from_image_pytesseract(image_path):
    import pytesseract
    img = Image.open(image_path)

    text = pytesseract.image_to_string(img)

    return text


def phi_ocr(image_url):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from PIL import Image
    import requests

    model_id = "yifeihu/TB-OCR-preview-0.1"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map=DEVICE, 
    trust_remote_code=True, 
    torch_dtype="auto", 
    # _attn_implementation='flash_attention_2',
    # torch_dtype=torch.float32 
    quantization_config=BitsAndBytesConfig(load_in_4bit=True) # Optional: Load model in 4-bit mode to save memory
    )

    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        num_crops=16,)

    question = "Return only the units in the format: Value,unit"
    # question = "Convert the text to markdown format." # this is required
    image = Image.open(requests.get(image_url, stream=True).raw)

    prompt_message = [{
        'role': 'user',
        'content': f'<|image_1|>\n{question}',
    }]
    prompt = processor.tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(DEVICE) 
    generation_args = { 
        "max_new_tokens": 1024, 
        "temperature": 0.1, 
        "do_sample": False
    }
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    response = response.split("<image_end>")[0] # remove the image_end token 

    return response

def text_from_image_tocr(image_path):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    image = Image.open(image_path)

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values,max_new_tokens=40,
    do_sample=True,
    top_p=0.92,
    top_k=0)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if __name__ == '__main__':
    
    response = phi_ocr('https://m.media-amazon.com/images/I/110EibNyclL.jpg')
    print(response)
    
    pass
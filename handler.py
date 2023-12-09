import subprocess


# Install required packages
commands = [
    'pip install -q git+https://github.com/huggingface/transformers.git',
    'pip install -q datasets sentencepiece',
    'pip install protobuf',
    'apt-get update -y',
    'apt-get install poppler-utils -y',
    'pip install pdf2image'
]

for command in commands:
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    if output:
        print('Output: ', output)
    if error:
        print('Error: ', error)

import runpod
import os
import time
import shutil
from pdf2image import convert_from_path
import torch
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_pdf(pdf_path, question):
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        pixel_values = processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        prompt = task_prompt.replace("{user_input}", question)
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        outputs = model.generate(pixel_values.to(device),
                                 decoder_input_ids=decoder_input_ids.to(device),
                                 max_length=model.decoder.config.max_position_embeddings,
                                 early_stopping=True,
                                 pad_token_id=processor.tokenizer.pad_token_id,
                                 eos_token_id=processor.tokenizer.eos_token_id,
                                 use_cache=True,
                                 num_beams=1,
                                 bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                 return_dict_in_generate=True,
                                 output_scores=True)

        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        answer = processor.token2json(seq)
        print(f"Answer for page {i}: {answer}")

def handler(event):
    print("pre event")
    print(event)

    question = event.get('input', {}).get('question')
    folder_path = event.get('input', {}).get('folder_path')

    if not question or not folder_path:
        print("Question or folder path not provided in event.")
        return "Error"

    print("IN PROGRESS:")
    
    # Iterate over all PDF files in the specified folder
    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Processing file: {pdf_path}")
            process_pdf(pdf_path, question)

    return "Processing complete"

runpod.serverless.start({
    "handler": handler
})

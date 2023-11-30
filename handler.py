import subprocess

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

    
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

    

def handler(event):
    print("pre event")
    print(event)

    # Extract folder_name from the event dictionary
    question = event.get('input', {}).get('question')
    # token_user = event.get('input', {}).get('token_user')
    # print("token_user:", token_user)
    if not question:
        print("question not provided in event.")
        return "Error"
    # if not token_user:
    #     print("Token user not provided in event.")
    #     return "Error"

    print("IN PROGRESS:")
    # do the things
    
    
 
    images = convert_from_path('1.pdf')
    
    for i in range(len(images)):   
        
        images[i].save('page'+ str(i) +'.jpg', 'JPEG')


    image=images[0]
    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(pixel_values.shape)

    

    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    #question = "Cual es el precio total?"
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
    print(seq)

    answer=processor.token2json(seq)

    #subprocess.run(["python", "codeServe_V3.py", '--audio_folder' ,folder_name, '--token_user', f'"{token_user}"'], check=True)

    ###########OPTIONS########################

    # Name of the audio file
    #audio_path = '13334_Gexa_npineros__FrontierMain__26__-1__+18179443510__2023-08-03_10-45-57.wav'
    # import sys
    # audio_path = sys.argv[1]
    # token_user = sys.argv[2]

    # s3_path = f"s3://gemma-middle-storage/{folder_name}"

    # subprocess.run(['aws', 's3', 'sync', s3_path, '.'], check=True)

    # pdf_files = get_files_with_extensions('.', ['.pdf'])

    # for pdf_file in pdf_files:
    #     audio_path = os.path.basename(pdf_file)
    #     print("Processing file: " + pdf_file)



    return answer

runpod.serverless.start({
    "handler": handler
})

import runpod
import os
import time
import subprocess
import shutil

import subprocess

commands = [
    'pip install -q git+https://github.com/huggingface/transformers.git',
    'pip install -q datasets sentencepiece',
    'pip install protobuf',
    'apt-get update',
    'apt-get install poppler-utils',
    'pip install pdf2image'
]

for command in commands:
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    if output:
        print('Output: ', output)
    if error:
        print('Error: ', error)







def handler(event):
    print("pre event")
    print(event)

    # Extract folder_name from the event dictionary
    folder_name = event.get('input', {}).get('folder')
    # token_user = event.get('input', {}).get('token_user')
    # print("token_user:", token_user)
    if not folder_name:
        print("Folder name not provided in event.")
        return "Error"
    # if not token_user:
    #     print("Token user not provided in event.")
    #     return "Error"

    print("IN PROGRESS:")
    # do the things
    
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



    return "Done"

runpod.serverless.start({
    "handler": handler
})

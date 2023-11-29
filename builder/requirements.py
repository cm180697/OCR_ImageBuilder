import subprocess

subprocess.run(["pip", "install", "runpod"])

subprocess.run(["pip", "install", "-q" , "git+https://github.com/huggingface/transformers.git" ])

subprocess.run(["pip", "install", "-q" , "datasets sentencepiece" ])

subprocess.run(["pip", "install", "-q" , "protobuf" ])

subprocess.run(["apt-get", "update" ])

subprocess.run(["apt-get", "install", "-y","poppler-utils" ])

subprocess.run(["pip", "install", "pdf2image" ])
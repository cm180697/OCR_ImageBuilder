import runpod
import os
import time
import subprocess

## load your model(s) into vram here
sleep_time = int(os.environ.get('SLEEP_TIME', 3))
subprocess.run(["python", "/appLLM/requirements.py"], check=True)
subprocess.run(["python", "/appLLM/modelLoader.py"], check=True)


def handler(event):
    print("pre event")
    print(event)

    # Extract folder_name from the event dictionary
    #folder_name = event.get('input', {}).get('folder')
    #if not folder_name:
    #    print("Folder name not provided in event.")
    #    return "Error"

    print("IN PROGRESS:")
    # do the things
    #subprocess.run(["python", "/appTrans/codeServe_V2.py", folder_name], check=True)
    print("DONE")
    
    return "Done"

runpod.serverless.start({
    "handler": handler
})

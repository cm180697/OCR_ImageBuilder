import torch
from pathlib import Path
import os
import subprocess

#Runpod
subprocess.run(["pip", "install", "runpod"])

# Check the current working directory
if Path.cwd().name != 'text-generation-webui':
    print("Installing the webui...")

    # Clone the git repository
    subprocess.run(["git", "clone", "https://github.com/oobabooga/text-generation-webui"])
    
    # Change the current working directory
    os.chdir('text-generation-webui')

    torver = torch.__version__
    print(f"TORCH: {torver}")
    is_cuda118 = '+cu118' in torver  # Example version format: 2.1.0+cu118
    is_cuda117 = '+cu117' in torver  # Example version format: 2.0.1+cu117

    # Read the requirements file
    textgen_requirements = open('requirements.txt').read().splitlines()
    if is_cuda117:
        textgen_requirements = [req.replace('+cu121', '+cu117').replace('+cu122', '+cu117').replace('torch2.1', 'torch2.0') for req in textgen_requirements]
    elif is_cuda118:
        textgen_requirements = [req.replace('+cu121', '+cu118').replace('+cu122', '+cu118') for req in textgen_requirements]
    
    # Write the modified requirements to a temporary file
    with open('temp_requirements.txt', 'w') as file:
        file.write('\n'.join(textgen_requirements))

    # Install the required packages using pip
    subprocess.run(["pip", "install", "-r", "extensions/api/requirements.txt", "--upgrade"])
    subprocess.run(["pip", "install", "-r", "temp_requirements.txt", "--upgrade"])

    print("\n --> If you see a warning about 'previously imported packages', just ignore it.")
    print("\n --> There is no need to restart the runtime.")

    # Try to import a module, if it fails, uninstall it using pip
    try:
        import flash_attn
    except ImportError:
        subprocess.run(["pip", "uninstall", "-y", "flash_attn"])
    
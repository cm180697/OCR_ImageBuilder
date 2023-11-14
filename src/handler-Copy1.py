""" Example handler file. """
import subprocess
import runpod

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
from transformers import AutoTokenizer, pipeline, logging,AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

model_name_or_path = "models/Llama-2-13b-chat-hf"
#model_basename = "asdasd"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token='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')

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
        #model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device_map="cuda:0",
        #use_triton=use_triton
        #quantize_config=None
        token='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'                    )


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    name = job_input.get('name', 'World')

    return f"Hello, {name}!"


runpod.serverless.start({"handler": handler})

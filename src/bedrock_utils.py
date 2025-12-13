import json
import logging
import time
import traceback

import boto3.session as boto3_session
import botocore.config

# ===================== NEW: IMPORT HUGGINGFACE LLAMA =====================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# ========================================================================


def extract_xml_tag(generation: str, tag):
    begin = generation.rfind(f"<{tag}>")
    if begin == -1:
        return
    begin = begin + len(f"<{tag}>")
    end = generation.rfind(f"</{tag}>", begin)
    if end == -1:
        return
    value = generation[begin:end].strip()
    return value


def predict_one_eg_mistral(x):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,
            connect_timeout=120,
            retries={"max_attempts": 5},
        ),
    )
    api_template = {
        "modelId": "mistral.mistral-7b-instruct-v0:2",
        "contentType": "application/json",
        "accept": "*/*",
        "body": "",
    }

    body = {
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 10,
        "prompt": x["prompt_input"],
    }

    api_template["body"] = json.dumps(body)

    for _ in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            response_body = json.loads(response.get("body").read())
            logging.info(response_body)
            return response_body["outputs"][0]["text"]
        except:
            traceback.print_exc()
            time.sleep(5)

    return ""


def predict_one_eg_claude_instant(x):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,
            connect_timeout=120,
            retries={"max_attempts": 5},
        ),
    )
    api_template = {
        "modelId": "anthropic.claude-instant-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": "",
    }

    body = {
        "max_tokens_to_sample": 512,
        "stop_sequences": ["\n\nHuman:"],
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 10,
        "prompt": "Human: {prompt}\nWrite your summary in <summary> XML tags.\n\nAssistant: ".format(
            prompt=x["prompt_input"].strip()
        ),
    }

    api_template["body"] = json.dumps(body)

    for _ in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            response_body = json.loads(response.get("body").read())
            summary = extract_xml_tag(response_body["completion"], "summary")
            logging.info(summary or response_body["completion"])
            return summary or response_body["completion"]
        except:
            traceback.print_exc()
            time.sleep(20)

    return ""


# ============== NEW: LLAMA-3.2-1B-INSTRUCT IMPLEMENTATION ==============

LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Caricamento tokenizer e modello UNA SOLA VOLTA (evita reload)
_llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
_llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
).to(_device)
_llama_model.eval()


@torch.no_grad()
def predict_one_eg_llama_light(x):
    """
    Llama-3.2-1B-Instruct su GPU Colab.
    Usa il prompt già pronto in x["prompt_input"] (prompts.py stile "Here is ...").
    Ritorna direttamente la completion (NO tag XML).
    """
    prompt = x["prompt_input"].strip()

    for _ in range(10):
        try:
            inputs = _llama_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(_device)

            output_ids = _llama_model.generate(
                **inputs,
                max_new_tokens=256,      # per summary corta basta spesso meno di 512
                do_sample=False,         # più stabile per summarization
                temperature=0.3,
                repetition_penalty=1.1,
                eos_token_id=_llama_tokenizer.eos_token_id,
                pad_token_id=_llama_tokenizer.eos_token_id,
            )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            completion = _llama_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            logging.info(completion)
            return completion

        except:
            traceback.print_exc()
            time.sleep(5)

    return ""



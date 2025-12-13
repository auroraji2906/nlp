import json
import logging
import time
import traceback

import boto3.session as boto3_session
import botocore.config


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

    success = False
    respone = None
    for _ in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            response_body = json.loads(response.get("body").read())
            logging.info(response_body)
            return response_body["outputs"][0]["text"]
        except:
            traceback.print_exc()
            time.sleep(5)

    if success:
        response_body = json.loads(response.get("body").read())
        logging.info(response_body)
        return response_body["outputs"][0]["text"]
    else:
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

    success = False
    response = None
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

    if success:
        response_body = json.loads(response.get("body").read())
        summary = extract_xml_tag(response_body["completion"], "summary")
        logging.info(summary or response_body["completion"])
        return summary or response_body["completion"]
    else:
        return ""


# ============== LLAMA-3.2-1B-INSTRUCT IMPLEMENTATION ==============

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
_llama_tokenizer = None
_llama_model = None

def _load_llama_once():
    global _llama_tokenizer, _llama_model
    if _llama_model is not None and _llama_tokenizer is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, use_fast=True)
    if _llama_tokenizer.pad_token_id is None:
        _llama_tokenizer.pad_token = _llama_tokenizer.eos_token

    # opzionale: 4-bit (se disponibile) per risparmiare VRAM
    try:
        _llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            load_in_4bit=True,
        )
    except Exception:
        _llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

    _llama_model.eval()


@torch.no_grad()
def predict_one_eg_llama_light(x):
    _load_llama_once()

    prompt = x["prompt_input"].strip()

    success = False
    response_body = None

    for _ in range(10):
        try:
            # ---- (1) wrap nel chat template SENZA cambiare i prompt.py ----
            # usa le stesse variabili: aggiorno solo la stringa "prompt"
            if hasattr(_llama_tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": prompt + "\n\nReturn ONLY the summary."},
                ]
                prompt = _llama_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            inputs = _llama_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            # ---- (2) device robusto con device_map="auto" ----
            target_device = next(_llama_model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            output_ids = _llama_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=_llama_tokenizer.eos_token_id,
                pad_token_id=_llama_tokenizer.pad_token_id,
            )

            # taglia il prompt -> output pulito (no domanda/testo)
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            text = _llama_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # ---- (3) pulizia leggera (senza cambiare prompt.py) ----
            # rimuove prefissi tipici che "sporcano" (optional ma utile)
            for p in ["Summary:", "TL;DR:", "Here is the summary:", "Here’s the summary:"]:
                if text.lower().startswith(p.lower()):
                    text = text[len(p):].strip()
                    break

            # crea “response_body” stile Bedrock/Mistral
            response_body = {"outputs": [{"text": text}]}
            success = True
            break

        except Exception:
            traceback.print_exc()
            time.sleep(5)

    if success:
        logging.info(response_body)
        return response_body["outputs"][0]["text"]
    else:
        return ""

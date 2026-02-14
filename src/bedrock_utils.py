import json
import logging
import os
import time
import traceback
from typing import Optional

import boto3.session as boto3_session
import botocore.config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_xml_tag(generation: str, tag: str) -> Optional[str]:
    begin = generation.rfind(f"<{tag}>")
    if begin == -1:
        return None
    begin = begin + len(f"<{tag}>")
    end = generation.rfind(f"</{tag}>", begin)
    if end == -1:
        return None
    return generation[begin:end].strip()


_BEDROCK_CLIENTS = {}


def _get_bedrock_client(region_name: str):
    if region_name in _BEDROCK_CLIENTS:
        return _BEDROCK_CLIENTS[region_name]

    current_session = boto3_session.Session()
    client = current_session.client(
        service_name="bedrock-runtime",
        region_name=region_name,
        endpoint_url=f"https://bedrock-runtime.{region_name}.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,
            connect_timeout=120,
            retries={"max_attempts": 5},
        ),
    )
    _BEDROCK_CLIENTS[region_name] = client
    return client


SONNET4_MODEL_ID_OR_ARN = os.getenv("SONNET4_MODEL_ID_OR_ARN", "").strip()
SONNET4_REGION = os.getenv("SONNET4_REGION", "eu-north-1").strip()

CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "512"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "1.0"))
CLAUDE_TOP_P = float(os.getenv("CLAUDE_TOP_P", "0.8"))

LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct").strip()
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3").strip()

_llama_tokenizer = None
_llama_model = None

_mistral_tokenizer = None
_mistral_model = None


def predict_one_eg_claude_sonnet4(x) -> str:
    if not SONNET4_MODEL_ID_OR_ARN:
        raise RuntimeError("Set SONNET4_MODEL_ID_OR_ARN environment variable.")

    bedrock = _get_bedrock_client(SONNET4_REGION)

    api_template = {
        "modelId": SONNET4_MODEL_ID_OR_ARN,
        "contentType": "application/json",
        "accept": "application/json",
        "body": "",
    }

    prompt_text = x["prompt_input"].strip()
    system_prompt = "You are an expert summarizer. Write the summary in <summary> XML tags."

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ],
        "max_tokens": CLAUDE_MAX_TOKENS,
        "temperature": CLAUDE_TEMPERATURE,
        "top_p": CLAUDE_TOP_P,
    }

    api_template["body"] = json.dumps(body)

    for _ in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            response_body = json.loads(response.get("body").read())

            completion_text = ""
            if "content" in response_body and response_body["content"]:
                completion_text = response_body["content"][0].get("text", "") or ""

            summary = extract_xml_tag(completion_text, "summary")
            result = summary if summary else completion_text

            logging.info(result)
            return result.strip()
        except Exception:
            traceback.print_exc()
            time.sleep(20)

    return ""


def _load_llama_once():
    global _llama_tokenizer, _llama_model
    if _llama_model is not None and _llama_tokenizer is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, use_fast=True)
    if _llama_tokenizer.pad_token_id is None:
        _llama_tokenizer.pad_token = _llama_tokenizer.eos_token

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


def _load_mistral_once():
    global _mistral_tokenizer, _mistral_model
    if _mistral_model is not None and _mistral_tokenizer is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME, use_fast=True)
    if _mistral_tokenizer.pad_token_id is None:
        _mistral_tokenizer.pad_token = _mistral_tokenizer.eos_token

    try:
        _mistral_model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            load_in_4bit=True,
        )
    except Exception:
        _mistral_model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

    _mistral_model.eval()


@torch.no_grad()
def predict_one_eg_llama_light(x) -> str:
    _load_llama_once()
    raw_prompt = x["prompt_input"].strip()

    for _ in range(10):
        try:
            prompt_for_model = raw_prompt

            if hasattr(_llama_tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": raw_prompt + "\n\nReturn ONLY the summary."},
                ]
                prompt_for_model = _llama_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            inputs = _llama_tokenizer(
                prompt_for_model,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

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

            gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            text = _llama_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            logging.info({"outputs": [{"text": text}]})
            return text
        except Exception:
            traceback.print_exc()
            time.sleep(5)

    return ""


@torch.no_grad()
def predict_one_eg_mistral_hf(x) -> str:
    _load_mistral_once()
    raw_prompt = x["prompt_input"].strip()

    for _ in range(10):
        try:
            prompt_for_model = raw_prompt

            if hasattr(_mistral_tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": raw_prompt + "\n\nReturn ONLY the summary."},
                ]
                prompt_for_model = _mistral_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            inputs = _mistral_tokenizer(
                prompt_for_model,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            target_device = next(_mistral_model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            output_ids = _mistral_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=_mistral_tokenizer.eos_token_id,
                pad_token_id=_mistral_tokenizer.pad_token_id,
            )

            gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            text = _mistral_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            logging.info({"outputs": [{"text": text}]})
            return text
        except Exception:
            traceback.print_exc()
            time.sleep(5)

    return ""

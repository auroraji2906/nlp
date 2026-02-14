#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import pathlib
from collections import defaultdict
from multiprocessing import Pool

import jsonlines
import nltk
import numpy as np
import tqdm
from bert_score import score as bertscore_score
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

from bedrock_utils import (
    predict_one_eg_claude_sonnet4,
    predict_one_eg_llama_light,
    predict_one_eg_mistral_hf,
)
from prompts import (
    ZS_EVIDENCE_PROMPT_STR_FOR_CLAUDE,
    ZS_EVIDENCE_PROMPT_STR_FOR_CLAUDE_ML,
    ZS_EVIDENCE_PROMPT_STR_FOR_LLAMA,
    ZS_EVIDENCE_PROMPT_STR_FOR_LLAMA_ML,
    ZS_EVIDENCE_PROMPT_STR_FOR_MISTRAL,
    ZS_EVIDENCE_PROMPT_STR_FOR_MISTRAL_ML,
    ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
    ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE_ML,
    ZS_KEYWORD_PROMPT_STR_FOR_LLAMA,
    ZS_KEYWORD_PROMPT_STR_FOR_LLAMA_ML,
    ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL_ML,
    ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
    ZS_NAIVE_PROMPT_STR_FOR_CLAUDE_ML,
    ZS_NAIVE_PROMPT_STR_FOR_LLAMA,
    ZS_NAIVE_PROMPT_STR_FOR_LLAMA_ML,
    ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    ZS_NAIVE_PROMPT_STR_FOR_MISTRAL_ML,
)

ZS_NAIVE_PROMPT_STR = {
    "mistral": ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
    "llama": ZS_NAIVE_PROMPT_STR_FOR_LLAMA,
}
ZS_KEYWORD_PROMPT_STR = {
    "mistral": ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
    "llama": ZS_KEYWORD_PROMPT_STR_FOR_LLAMA,
}
ZS_EVIDENCE_PROMPT_STR = {
    "mistral": ZS_EVIDENCE_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_EVIDENCE_PROMPT_STR_FOR_CLAUDE,
    "llama": ZS_EVIDENCE_PROMPT_STR_FOR_LLAMA,
}

ZS_NAIVE_PROMPT_STR_ML = {
    "mistral": ZS_NAIVE_PROMPT_STR_FOR_MISTRAL_ML,
    "claude": ZS_NAIVE_PROMPT_STR_FOR_CLAUDE_ML,
    "llama": ZS_NAIVE_PROMPT_STR_FOR_LLAMA_ML,
}
ZS_KEYWORD_PROMPT_STR_ML = {
    "mistral": ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL_ML,
    "claude": ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE_ML,
    "llama": ZS_KEYWORD_PROMPT_STR_FOR_LLAMA_ML,
}
ZS_EVIDENCE_PROMPT_STR_ML = {
    "mistral": ZS_EVIDENCE_PROMPT_STR_FOR_MISTRAL_ML,
    "claude": ZS_EVIDENCE_PROMPT_STR_FOR_CLAUDE_ML,
    "llama": ZS_EVIDENCE_PROMPT_STR_FOR_LLAMA_ML,
}


def _is_english(target_lang: str | None) -> bool:
    if target_lang is None:
        return True
    x = target_lang.strip().lower()
    return x in {"", "en", "eng", "english"}


def estimate_logits_threshold(dataset_file, percentile_threshold):
    if not os.path.exists(dataset_file):
        logging.warning("validation set not found for logits threshold.")
        return -1

    with jsonlines.open(dataset_file) as f:
        data = list(f)

    if not data or "input_kw_model" not in data[0]:
        logging.warning("input_kw_model not found in the file. Use -1 as threshold.")
        return -1

    logits = []
    for item in data:
        for kw_info_model in item.get("input_kw_model", []):
            logits.append(kw_info_model["score"])

    if not logits:
        logging.warning("No logits found. Use -1 as threshold.")
        return -1

    return float(np.percentile(logits, percentile_threshold))


def remove_duplicate_top_k(candidates, top_k, threshold=70):
    ret = []
    for candidate in candidates:
        to_delete = set()
        to_skip = False

        if len(ret) >= top_k:
            break

        for added_kw in ret:
            if fuzz.ratio(added_kw["phrase"].lower(), candidate["phrase"].lower()) >= threshold:
                if len(added_kw["phrase"]) <= len(candidate["phrase"]):
                    to_delete.add(added_kw["phrase"])
                else:
                    to_skip = True

        ret = [item for item in ret if item["phrase"] not in to_delete]

        if not to_skip:
            ret.append(candidate)

    return ret


class NaivePrompt(object):
    def __init__(self, prompt: str, target_lang: str | None = None):
        self.prompt = prompt
        self.target_lang = target_lang

    def __call__(self, example):
        p = self.prompt.replace("<text>", example["trunc_input"])
        if self.target_lang:
            p = p.replace("<target_language>", self.target_lang)
        return p


class SegExtTopK(object):
    def __init__(
        self,
        prompt: str,
        top_k: int,
        deduplicate: bool = True,
        logits_threshold: float = -1,
        use_rank: bool = False,
        target_lang: str | None = None,
    ):
        self.prompt = prompt
        self.top_k = top_k
        self.deduplicate = deduplicate
        self.logits_threshold = logits_threshold
        self.use_rank = use_rank
        self.target_lang = target_lang

    def __call__(self, example):
        if self.use_rank:
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            for i in range(len(selected_keywords)):
                selected_keywords[i]["score"] = i
        else:
            selected_keywords = []
            for kw_info in sorted(example["input_kw_model"], key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                selected_keywords.append(example["trunc_input_phrases"][kw_info["kw_index"]])
                selected_keywords[-1]["score"] = kw_info["score"]

        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=self.top_k)
        else:
            selected_keywords = selected_keywords[: self.top_k]

        formatted_keywords = "; ".join([item["phrase"] for item in selected_keywords]) + "."
        p = self.prompt.replace("<text>", example["trunc_input"]).replace("<keywords>", formatted_keywords)
        if self.target_lang:
            p = p.replace("<target_language>", self.target_lang)
        return p


class GraphSigExtPrompt(object):
    def __init__(
        self,
        prompt: str,
        kw_model_top_k: int,
        logits_threshold: float = -1,
        terminals: int = 10,
        sent_budget: int = 10,
        window: int = 2,
        match_threshold: int = 70,
        deduplicate: bool = True,
        use_rank: bool = False,
        target_lang: str | None = None,
    ):
        self.prompt = prompt
        self.kw_model_top_k = kw_model_top_k
        self.logits_threshold = logits_threshold
        self.terminals = terminals
        self.sent_budget = sent_budget
        self.window = window
        self.match_threshold = match_threshold
        self.deduplicate = deduplicate
        self.use_rank = use_rank
        self.target_lang = target_lang

    def _match_kw_sent(self, kw_phrase, sent):
        return fuzz.partial_ratio(kw_phrase.lower(), sent.lower()) >= self.match_threshold

    def _select_keywords(self, example):
        if self.use_rank:
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            for i in range(len(selected_keywords)):
                selected_keywords[i]["score"] = i
        else:
            selected_keywords = []
            for kw_info in sorted(example["input_kw_model"], key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                selected_keywords.append(example["trunc_input_phrases"][kw_info["kw_index"]])
                selected_keywords[-1]["score"] = kw_info["score"]

        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=self.kw_model_top_k)
        else:
            selected_keywords = selected_keywords[: self.kw_model_top_k]
        return selected_keywords

    def _build_adj(self, sentences, kw_phrases):
        adj = defaultdict(list)
        n = len(sentences)

        for i in range(n):
            for d in range(1, self.window + 1):
                if i + d < n:
                    a = ("S", i)
                    b = ("S", i + d)
                    adj[a].append(b)
                    adj[b].append(a)

        for j, kw in enumerate(kw_phrases):
            kn = ("K", j)
            for i, s in enumerate(sentences):
                if self._match_kw_sent(kw, s):
                    sn = ("S", i)
                    adj[kn].append(sn)
                    adj[sn].append(kn)

        return adj

    def _build_personalization(self, terminal_phrases, sentences):
        pers = {}
        for j in range(len(terminal_phrases)):
            pers[("K", j)] = 1.0
        for i, s in enumerate(sentences):
            if any(self._match_kw_sent(kw, s) for kw in terminal_phrases):
                pers[("S", i)] = pers.get(("S", i), 0.0) + 0.25
        return pers

    def _pagerank(self, adj, nodes, personalization, damping=0.85, max_iter=100, tol=1e-8):
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}

        outdeg = np.zeros(n, dtype=np.float64)
        neigh = [[] for _ in range(n)]
        for u in nodes:
            ui = idx[u]
            vs = adj.get(u, [])
            neigh[ui] = [idx[v] for v in vs if v in idx]
            outdeg[ui] = len(neigh[ui])

        p = np.zeros(n, dtype=np.float64)
        for node, w in personalization.items():
            if node in idx:
                p[idx[node]] += float(w)
        s = p.sum()
        if s <= 0:
            p[:] = 1.0 / n
        else:
            p /= s

        r = p.copy()

        for _ in range(max_iter):
            r_new = np.zeros(n, dtype=np.float64)

            dangling_mass = 0.0
            for ui in range(n):
                if outdeg[ui] == 0:
                    dangling_mass += r[ui]
                else:
                    share = r[ui] / outdeg[ui]
                    for vi in neigh[ui]:
                        r_new[vi] += share

            r_new = damping * (r_new + dangling_mass * p) + (1 - damping) * p

            if np.linalg.norm(r_new - r, 1) < tol:
                r = r_new
                break
            r = r_new

        return {nodes[i]: float(r[i]) for i in range(n)}

    def __call__(self, example):
        selected_keywords = self._select_keywords(example)
        terminal_kws = selected_keywords[: min(self.terminals, len(selected_keywords))]
        terminal_phrases = [k["phrase"] for k in terminal_kws]

        sentences = nltk.sent_tokenize(example["trunc_input"])
        if not sentences:
            sentences = [example["trunc_input"]]

        adj = self._build_adj(sentences, terminal_phrases)
        nodes = [("S", i) for i in range(len(sentences))] + [("K", j) for j in range(len(terminal_phrases))]

        personalization = self._build_personalization(terminal_phrases, sentences)
        scores = self._pagerank(adj, nodes, personalization)

        sent_scores = [(i, scores.get(("S", i), 0.0)) for i in range(len(sentences))]
        sent_scores.sort(key=lambda x: x[1], reverse=True)

        top_ids = [i for i, _ in sent_scores[: min(self.sent_budget, len(sent_scores))]]
        top_ids = sorted(set(top_ids))
        picked = [sentences[i] for i in top_ids]

        kw_str = "; ".join(terminal_phrases).strip()
        ev_str = " ".join(picked).strip()

        if kw_str and not kw_str.endswith("."):
            kw_str += "."
        if ev_str and not ev_str.endswith("."):
            ev_str += "."

        p = (
            self.prompt.replace("<text>", example["trunc_input"])
            .replace("<keywords>", kw_str)
            .replace("<evidence>", ev_str)
        )
        if self.target_lang:
            p = p.replace("<target_language>", self.target_lang)
        return p


def get_prompt_fn(model_name, dataset, kw_strategy, kw_model_top_k, logits_threshold, target_lang, args):
    target_language = target_lang.strip() or None
    multilingual_mode = not _is_english(target_language)

    if multilingual_mode:
        naive_prompt = ZS_NAIVE_PROMPT_STR_ML[model_name][dataset]
        kw_prompt = ZS_KEYWORD_PROMPT_STR_ML[model_name][dataset]
        ev_prompt = ZS_EVIDENCE_PROMPT_STR_ML[model_name][dataset]
        tl = target_language
    else:
        naive_prompt = ZS_NAIVE_PROMPT_STR[model_name][dataset]
        kw_prompt = ZS_KEYWORD_PROMPT_STR[model_name][dataset]
        ev_prompt = ZS_EVIDENCE_PROMPT_STR[model_name][dataset]
        tl = None

    if kw_strategy == "disable":
        return NaivePrompt(naive_prompt, target_lang=tl)
    elif kw_strategy == "sigext_topk":
        return SegExtTopK(
            kw_prompt,
            top_k=kw_model_top_k,
            logits_threshold=logits_threshold,
            target_lang=tl,
        )
    elif kw_strategy == "graph_sigext":
        return GraphSigExtPrompt(
            ev_prompt,
            kw_model_top_k=kw_model_top_k,
            logits_threshold=logits_threshold,
            terminals=getattr(args, "graph_terminals", 10),
            sent_budget=getattr(args, "graph_sent_budget", 10),
            window=getattr(args, "graph_window", 2),
            match_threshold=getattr(args, "graph_match_threshold", 70),
            deduplicate=True,
            use_rank=False,
            target_lang=tl,
        )
    else:
        raise RuntimeError("unknown kw strategy.")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_rouge_score(inference_data, preds):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    labels = [item["raw_output"] for item in inference_data]
    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result_element = defaultdict(list)
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(target=label, prediction=pred)
        for metric_name, value in score.items():
            result_element[f"{metric_name}p"].append(value.precision)
            result_element[f"{metric_name}r"].append(value.recall)
            result_element[f"{metric_name}f"].append(value.fmeasure)

    result = {k: float(np.mean(v)) for k, v in result_element.items()}
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [len(word_tokenize(pred)) for pred in preds]
    result["gen_len"] = float(np.mean(prediction_lens))
    return result


def compute_bertscore(inference_data, preds, model_type, device="cpu"):
    refs = [item["raw_output"] for item in inference_data]
    preds = [p.strip() for p in preds]
    refs = [r.strip() for r in refs]

    P, R, F1 = bertscore_score(
        preds,
        refs,
        model_type=model_type,
        device=device,
        rescale_with_baseline=False,
        lang=None,
    )
    return {
        "bertscore_p": round(float(P.mean().item()) * 100.0, 4),
        "bertscore_r": round(float(R.mean().item()) * 100.0, 4),
        "bertscore_f1": round(float(F1.mean().item()) * 100.0, 4),
        "bertscore_model": model_type,
    }


def run_inference(
    model_name,
    kw_strategy,
    kw_model_top_k,
    dataset,
    dataset_dir,
    output_dir,
    inference_on_split="test",
    target_lang="en",
    bertscore_model="roberta-large",
    bertscore_device="cuda",
    graph_sent_budget=10,
    graph_window=2,
    graph_match_threshold=70,
    graph_terminals=10,
):
    dataset_dir = pathlib.Path(dataset_dir)
    logits_threshold = estimate_logits_threshold(str(dataset_dir.joinpath("validation.jsonl")), 75)
    logging.info(f"logits threshold is {logits_threshold}")

    if model_name == "mistral":
        predict_one_eg_fn = predict_one_eg_mistral_hf
    elif model_name == "claude":
        predict_one_eg_fn = predict_one_eg_claude_sonnet4
    elif model_name == "llama":
        predict_one_eg_fn = predict_one_eg_llama_light
    else:
        raise ValueError(f"invalid model name {model_name}")

    graph_args = argparse.Namespace(
        graph_sent_budget=graph_sent_budget,
        graph_window=graph_window,
        graph_match_threshold=graph_match_threshold,
        graph_terminals=graph_terminals,
    )

    prompting_fn = get_prompt_fn(
        model_name=model_name,
        dataset=dataset,
        kw_strategy=kw_strategy,
        kw_model_top_k=kw_model_top_k,
        logits_threshold=logits_threshold,
        target_lang=target_lang,
        args=graph_args,
    )

    dataset_filename = str(dataset_dir.joinpath(f"{inference_on_split}.jsonl"))
    with jsonlines.open(dataset_filename) as f:
        inference_data = list(f)

    with Pool(4) as p:
        all_prompt = list(tqdm.tqdm(p.imap(prompting_fn, inference_data), total=len(inference_data)))

    for i in range(len(inference_data)):
        inference_data[i]["prompt_input"] = all_prompt[i]

    if model_name in {"llama", "mistral"}:
        all_res = [predict_one_eg_fn(item) for item in tqdm.tqdm(inference_data, total=len(inference_data))]
    else:
        with Pool(4) as p:
            all_res = list(tqdm.tqdm(p.imap(predict_one_eg_fn, inference_data), total=len(inference_data)))

    output_path = str(pathlib.Path(output_dir).expanduser())
    os.makedirs(output_path, exist_ok=True)

    with jsonlines.open(output_path + f"/{inference_on_split}_dataset.jsonl", "w") as f:
        f.write_all(inference_data)

    with open(output_path + f"/{inference_on_split}_predictions.json", "w") as f:
        json.dump(all_res, f, indent=2)

    target_language = target_lang.strip() or None
    multilingual_mode = not _is_english(target_language)

    if multilingual_mode and bertscore_model.strip() == "roberta-large":
        effective_bertscore_model = "xlm-roberta-large"
    else:
        effective_bertscore_model = bertscore_model.strip()

    test_metrics = {}
    if not multilingual_mode:
        test_metrics.update(compute_rouge_score(inference_data, all_res))
    else:
        logging.info("Multilingual mode active: skipping ROUGE (no gold summaries in target language).")
        prediction_lens = [len(word_tokenize(pred)) for pred in all_res]
        test_metrics["gen_len"] = float(np.mean(prediction_lens))
        test_metrics["target_lang"] = target_language or ""

    test_metrics.update(
        compute_bertscore(inference_data, all_res, model_type=effective_bertscore_model, device=bertscore_device)
    )

    with open(str(pathlib.Path(output_dir).joinpath(f"{inference_on_split}_metrics.json")), "w") as f:
        json.dump(test_metrics, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="llama", choices=["claude", "mistral", "llama"], help="llm name")
    parser.add_argument(
        "--kw_strategy",
        required=True,
        choices=["disable", "sigext_topk", "graph_sigext"],
        help="keyword strategy.",
    )
    parser.add_argument("--kw_model_top_k", default=20, type=int, help="top-k keyphrases used by sigext/graph.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxiv", "pubmed", "cnn", "samsum", "meetingbank"],
        help="Select from supported datasets.",
    )
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train/validation/test jsonl.")
    parser.add_argument("--output_dir", required=True, type=str, help="directory to save experiment.")
    parser.add_argument("--inference_on_split", default="test", type=str, help="split_to_run_inference")

    parser.add_argument("--target_lang", default="en", type=str, help="Target language for generated summaries.")
    parser.add_argument("--bertscore_model", default="roberta-large", type=str, help="Default: roberta-large (English).")
    parser.add_argument("--bertscore_device", default="cuda", choices=["cpu", "cuda"], help="Default: cuda.")

    parser.add_argument("--graph_sent_budget", default=10, type=int, help="max #sentences used as evidence.")
    parser.add_argument("--graph_window", default=2, type=int, help="sentence adjacency window.")
    parser.add_argument("--graph_match_threshold", default=70, type=int, help="fuzzy match threshold (0-100).")
    parser.add_argument("--graph_terminals", default=10, type=int, help="how many top keyphrases as terminals.")

    args = parser.parse_args()
    run_inference(**vars(args))


if __name__ == "__main__":
    main()

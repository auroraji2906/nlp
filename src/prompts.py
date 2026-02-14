# =========================
# NAIVE PROMPTS
# =========================

ZS_NAIVE_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
           "Please write a short summary for the article in 1-2 sentences.[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
             "Please write a short abstract in about 3 sentences.[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
              "Please write a short abstract in about 3 sentences.[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n"
              "Please write a short 1 sentence summary.[/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n"
                   "Please write a 2-3 sentence summary.[/INST]",
}

ZS_NAIVE_PROMPT_STR_FOR_CLAUDE = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a summary for the article in 2-3 sentences.",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a comprehensive paper abstract section.",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a paper abstract in around 5 sentences.",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary.",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a summary in about 5 sentences.",
}

ZS_NAIVE_PROMPT_STR_FOR_LLAMA = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary (3-5 sentences).",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise summary (3-5 sentences).",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise summary (3-5 sentences).",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary.",
    "meetingbank": "Here is conversation:\n<text>\n\n"
                   "Please write a concise summary (4-6 sentences).",
}


# =========================
# KEYWORD PROMPTS
# =========================

ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
           "Please write a short summary for the article in 1-2 sentences. "
           "Consider including the following information: <keywords>[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
             "Please write a short abstract in about 3 sentences. "
             "Consider including the following information: <keywords>[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
              "Please write a short abstract in about 3 sentences. "
              "Consider including the following information: <keywords>[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n"
              "Please write a short 1 sentence summary. "
              "Consider including the following information: <keywords>[/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n"
                   "Please write a 2-3 sentence summary. "
                   "Consider including the following information: <keywords>[/INST]",
}

ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a summary for the article in 2-3 sentences. "
           "Consider including the following information: <keywords>.",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a comprehensive paper abstract section. "
             "Consider including the following information: <keywords>.",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a paper abstract in around 5 sentences. "
              "Consider including the following information: <keywords>.",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary. "
              "Consider including the following information: <keywords>.",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a summary in about 5 sentences. "
                   "Consider including the following information: <keywords>.",
}

ZS_KEYWORD_PROMPT_STR_FOR_LLAMA = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary (3-5 sentences). "
           "Consider including the following information: <keywords>",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise summary (3-5 sentences). "
             "Consider including the following information: <keywords>.",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise summary (3-5 sentences). "
              "Consider including the following information: <keywords>.",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary. "
              "Consider including the following information: <keywords>.",
    "meetingbank": "Here is conversation:\n<text>\n\n"
                   "Please write a concise summary (4-6 sentences). "
                   "Consider including the following information: <keywords>.",
}


# =========================
# EVIDENCE PROMPTS
# =========================

ZS_EVIDENCE_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
           "Please write a short summary in 3-5 sentences.\n"
           "Consider including the following key information: <keywords>\n"
           "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
             "Please write a concise abstract in 3-5 sentences.\n"
             "Consider including the following key information: <keywords>\n"
             "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
              "Please write a concise abstract in 3-5 sentences.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n"
                   "Please write a concise summary in 4-6 sentences.\n"
                   "Consider including the following key information: <keywords>\n"
                   "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
}

ZS_EVIDENCE_PROMPT_STR_FOR_CLAUDE = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary in 3-5 sentences.\n"
           "Consider including the following key information: <keywords>\n"
           "You may use the following supporting sentences as evidence:\n<evidence>",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise abstract in 3-5 sentences.\n"
             "Consider including the following key information: <keywords>\n"
             "You may use the following supporting sentences as evidence:\n<evidence>",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise abstract in 3-5 sentences.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a concise summary in 4-6 sentences.\n"
                   "Consider including the following key information: <keywords>\n"
                   "You may use the following supporting sentences as evidence:\n<evidence>",
}

ZS_EVIDENCE_PROMPT_STR_FOR_LLAMA = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary (3-5 sentences).\n"
           "Consider including the following information: <keywords>.\n"
           "Consider using the following supporting sentences: <evidence>",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise summary (3-5 sentences).\n"
             "Consider including the following information: <keywords>.\n"
             "Consider using the following supporting sentences: <evidence>",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise summary (3-5 sentences).\n"
              "Consider including the following information: <keywords>.\n"
              "Consider using the following supporting sentences: <evidence>",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary.\n"
              "Consider including the following information: <keywords>.\n"
              "Consider using the following supporting sentences: <evidence>",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a concise summary (4-6 sentences).\n"
                   "Consider including the following information: <keywords>.\n"
                   "Consider using the following supporting sentences: <evidence>",
}


# =========================
# MULTILINGUAL PROMPTS (same style; only adds "in <target_language>")
# =========================

ZS_NAIVE_PROMPT_STR_FOR_MISTRAL_ML = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
           "Please write a short summary for the article in <target_language> in 1-2 sentences.[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
             "Please write a short abstract in <target_language> in about 3 sentences.[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
              "Please write a short abstract in <target_language> in about 3 sentences.[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n"
              "Please write a short 1 sentence summary in <target_language>.[/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n"
                   "Please write a 2-3 sentence summary in <target_language>.[/INST]",
}

ZS_NAIVE_PROMPT_STR_FOR_CLAUDE_ML = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a summary for the article in <target_language> in 2-3 sentences.",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a comprehensive paper abstract section in <target_language>.",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a paper abstract in <target_language> in around 5 sentences.",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary in <target_language>.",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a summary in <target_language> in about 5 sentences.",
}

ZS_NAIVE_PROMPT_STR_FOR_LLAMA_ML = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary in <target_language> (3-5 sentences).",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise summary in <target_language> (3-5 sentences).",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise summary in <target_language> (3-5 sentences).",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary in <target_language>.",
    "meetingbank": "Here is conversation:\n<text>\n\n"
                   "Please write a concise summary in <target_language> (4-6 sentences).",
}

ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL_ML = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
           "Please write a short summary for the article in <target_language> in 1-2 sentences. "
           "Consider including the following information: <keywords>[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
             "Please write a short abstract in <target_language> in about 3 sentences. "
             "Consider including the following information: <keywords>[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
              "Please write a short abstract in <target_language> in about 3 sentences. "
              "Consider including the following information: <keywords>[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n"
              "Please write a short 1 sentence summary in <target_language>. "
              "Consider including the following information: <keywords>[/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n"
                   "Please write a 2-3 sentence summary in <target_language>. "
                   "Consider including the following information: <keywords>[/INST]",
}

ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE_ML = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a summary for the article in <target_language> in 2-3 sentences. "
           "Consider including the following information: <keywords>.",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a comprehensive paper abstract section in <target_language>. "
             "Consider including the following information: <keywords>.",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a paper abstract in <target_language> in around 5 sentences. "
              "Consider including the following information: <keywords>.",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary in <target_language>. "
              "Consider including the following information: <keywords>.",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a summary in <target_language> in about 5 sentences. "
                   "Consider including the following information: <keywords>.",
}

ZS_KEYWORD_PROMPT_STR_FOR_LLAMA_ML = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary in <target_language> (3-5 sentences).\n"
           "Consider including the following information: <keywords>",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise summary in <target_language> (3-5 sentences).\n"
             "Consider including the following information: <keywords>.",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise summary in <target_language> (3-5 sentences).\n"
              "Consider including the following information: <keywords>.",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary in <target_language>.\n"
              "Consider including the following information: <keywords>.",
    "meetingbank": "Here is conversation:\n<text>\n\n"
                   "Please write a concise summary in <target_language> (4-6 sentences).\n"
                   "Consider including the following information: <keywords>.",
}


# =========================
# MULTILINGUAL + EVIDENCE PROMPTS
# =========================

ZS_EVIDENCE_PROMPT_STR_FOR_MISTRAL_ML = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
           "Please write a short summary in <target_language> in 3-5 sentences.\n"
           "Consider including the following key information: <keywords>\n"
           "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
             "Please write a concise abstract in <target_language> in 3-5 sentences.\n"
             "Consider including the following key information: <keywords>\n"
             "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
              "Please write a concise abstract in <target_language> in 3-5 sentences.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary in <target_language>.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n"
                   "Please write a concise summary in <target_language> in 4-6 sentences.\n"
                   "Consider including the following key information: <keywords>\n"
                   "You may use the following supporting sentences as evidence:\n<evidence>[/INST]",
}

ZS_EVIDENCE_PROMPT_STR_FOR_CLAUDE_ML = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary in <target_language> in 3-5 sentences.\n"
           "Consider including the following key information: <keywords>\n"
           "You may use the following supporting sentences as evidence:\n<evidence>",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise abstract in <target_language> in 3-5 sentences.\n"
             "Consider including the following key information: <keywords>\n"
             "You may use the following supporting sentences as evidence:\n<evidence>",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise abstract in <target_language> in 3-5 sentences.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary in <target_language>.\n"
              "Consider including the following key information: <keywords>\n"
              "You may use the following supporting sentences as evidence:\n<evidence>",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a concise summary in <target_language> in 4-6 sentences.\n"
                   "Consider including the following key information: <keywords>\n"
                   "You may use the following supporting sentences as evidence:\n<evidence>",
}

ZS_EVIDENCE_PROMPT_STR_FOR_LLAMA_ML = {
    "cnn": "Here is a news article:\n<text>\n\n"
           "Please write a short summary in <target_language> (3-5 sentences).\n"
           "Consider including the following information: <keywords>.\n"
           "Consider using the following supporting sentences: <evidence>",
    "arxiv": "Here is a research paper:\n<text>\n\n"
             "Please write a concise summary in <target_language> (3-5 sentences).\n"
             "Consider including the following information: <keywords>.\n"
             "Consider using the following supporting sentences: <evidence>",
    "pubmed": "Here is a research paper:\n<text>\n\n"
              "Please write a concise summary in <target_language> (3-5 sentences).\n"
              "Consider including the following information: <keywords>.\n"
              "Consider using the following supporting sentences: <evidence>",
    "samsum": "Here is a conversation:\n<text>\n\n"
              "Please write a very short 1 sentence summary in <target_language>.\n"
              "Consider including the following information: <keywords>.\n"
              "Consider using the following supporting sentences: <evidence>",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
                   "Please write a concise summary in <target_language> (4-6 sentences).\n"
                   "Consider including the following information: <keywords>.\n"
                   "Consider using the following supporting sentences: <evidence>",
}

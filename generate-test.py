#!/usr/bin/env python

from transformers import AutoTokenizer

VARIANTS_TO_TEST = [
    'ibm-granite/granite-3.1-8b-instruct'
]

HISTORY = [
    { 'role': 'system', 'content': 'You are a helpful assistant' },
    { 'role': 'user', 'content': 'Hello' },
    { 'role': 'assistant', 'content': 'Hi there' },
    { 'role': 'user', 'content': 'Who are you' },
    { 'role': 'assistant', 'content': '   I am an assistant   ' },
    { 'role': 'user', 'content': 'Another question' },
]

for variant in VARIANTS_TO_TEST:
    history = [m for m in HISTORY] # copy
    if 'Mistral' in variant or 'gemma' in variant:
        history.pop(0) # no system prompt for mistral and gemma
    if 'gemma' in variant:
        # GemmaTokenizer is quite buggy, let's hard code the template here
        GEMMA_TMLP = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
        print("\n----- Gemma -----")
        output = AutoTokenizer.from_pretrained(VARIANTS_TO_TEST[0]).apply_chat_template(history, tokenize=False, add_generation_prompt=True, chat_template=GEMMA_TMLP)
        print(output)
        print("\n[Test String]\n// google/gemma-7b-it")
        print(output.replace("\n", "\\n"))
        print('"' + output.replace("\n", "\\n") + '",')
    else:
        print("\n----- " + variant + " -----")
        tokenizer = AutoTokenizer.from_pretrained(variant)
        output = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        print(output)
        print("\n[Test String]\n// " + variant)
        print('"' + output.replace("\n", "\\n") + '",')
import re

import torch

from .language import detect_language
from .model_loader import Translator

_TAG_RE = re.compile(r"<[^>]+>")


def _build_llm_prompt(tokenizer, system_prompt, text):
    src = detect_language(text)

    candidates = [
        [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": src,
                "target_lang_code": "tr",
                "text": text,
            }],
        }],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        [{"role": "user", "content": f"{system_prompt}\n\n{text}"}],
    ]

    for messages in candidates:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            continue

    return (
        f"<start_of_turn>user\n{system_prompt}\n\n{text}"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )


def translate_in_batches(
    tr: Translator,
    texts,
    system_prompt,
    batch_size,
    max_new_tokens,
    num_beams,
):
    if not texts:
        return []

    if tr.model_kind in ("nllb", "seq2seq"):
        return _translate_seq2seq(tr, texts, batch_size, max_new_tokens, num_beams)

    if tr.model_kind == "multimodal":
        return _translate_multimodal(tr, texts, max_new_tokens, num_beams)

    return _translate_llm(tr, texts, system_prompt, max_new_tokens, num_beams)


def _translate_seq2seq(tr, texts, batch_size, max_new_tokens, num_beams):
    results = []
    gen_kwargs = {"max_length": max_new_tokens, "num_beams": num_beams}
    if tr.forced_bos_id is not None:
        gen_kwargs["forced_bos_token_id"] = tr.forced_bos_id

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        encoded = tr.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        encoded = {k: v.to(tr.device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = tr.model.generate(**encoded, **gen_kwargs)

        results.extend(
            tr.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated
        )
    return results


def _translate_multimodal(tr, texts, max_new_tokens, num_beams):
    results = []
    for text in texts:
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": detect_language(text),
                "target_lang_code": "tr",
                "text": text,
            }],
        }]
        inputs = tr.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_len = inputs["input_ids"].shape[-1]
        inputs = {k: v.to(tr.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = tr.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

        new_tokens = output[0][input_len:]
        decoded = tr.tokenizer.decode(new_tokens, skip_special_tokens=False)
        decoded = decoded.split("<end_of_turn>")[0]
        decoded = _TAG_RE.sub("", decoded)
        results.append(decoded.strip())
    return results


def _translate_llm(tr, texts, system_prompt, max_new_tokens, num_beams):
    results = []
    for text in texts:
        prompt = _build_llm_prompt(tr.tokenizer, system_prompt, text)
        inputs = tr.tokenizer(prompt, return_tensors="pt").to(tr.device)

        with torch.no_grad():
            output = tr.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

        prompt_len = inputs.input_ids.shape[-1]
        results.append(
            tr.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
        )
    return results

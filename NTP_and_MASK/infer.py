import os
import random
import re

import numpy as np
import torch
from tqdm import tqdm

from utils.get_embedding import LM
from utils.loader import DatasetManager
from utils.metrics import BERTScore, BLEUScore, Statistic
from utils.test_config import args


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def strip_ds(text):
    text = re.sub(r"</?ds>", "", text)
    text = re.sub(r"/?ds", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_interjections(arg_interjection):
    if isinstance(arg_interjection, (list, tuple)):
        phrases = list(arg_interjection)
    else:
        normalized = str(arg_interjection)
        for sep in ["|", "；", ";"]:
            normalized = normalized.replace(sep, ",")
        phrases = [item.strip() for item in normalized.split(",") if item.strip()]

    deduped = []
    seen = set()
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            deduped.append(phrase)
    return deduped


def tokenize_phrases(tokenizer, phrases):
    phrase_ids_list = []
    idseq_to_phrase = {}
    for phrase in phrases:
        encoded = tokenizer(phrase, add_special_tokens=False, return_tensors="pt")
        token_ids = encoded["input_ids"][0].tolist()
        if token_ids:
            phrase_ids_list.append(token_ids)
            idseq_to_phrase[tuple(token_ids)] = phrase
    return phrase_ids_list, idseq_to_phrase


@torch.no_grad()
def infer_with_phrase_stats(prompt, groundtruth, model, tokenizer, target_phrases):
    device = model.device
    eos_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    reference = tokenizer(groundtruth, return_tensors="pt", truncation=True, max_length=1024).to(device)
    max_new_tokens = max(int(reference["input_ids"].shape[1]), 1)

    phrase_ids_list, idseq_to_phrase = tokenize_phrases(tokenizer, target_phrases)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    past_key_values = None

    generated_ids = []
    phrase_hits = []

    for step in range(1, max_new_tokens + 1):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits_last = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        log_probs = torch.log_softmax(logits_last[0], dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum())
        ppl_t = float(torch.exp(torch.tensor(entropy)))

        sorted_logits, sorted_idx = torch.sort(logits_last[0], descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > 0.9
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        filtered_logits = sorted_logits.masked_fill(mask, float("-inf"))
        next_idx = torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1)
        next_token = sorted_idx[next_idx]

        token_id = int(next_token.item())
        generated_ids.append(token_id)

        for ids in phrase_ids_list:
            length = len(ids)
            if len(generated_ids) >= length and generated_ids[-length:] == ids:
                phrase_hits.append(
                    {
                        "step_end": step,
                        "phrase": idseq_to_phrase[tuple(ids)],
                        "ppl_instant": ppl_t,
                    }
                )

        input_ids = next_token.view(1, 1)
        attention_mask = None
        if token_id == eos_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True), phrase_hits


if __name__ == "__main__":
    set_seed(42)

    save_base_path = f"./infer_result/{args.pretrainModel}/{args.language}+{args.test_type}+{args.infer_ratio}+"
    os.makedirs(f"./infer_result/{args.pretrainModel}/", exist_ok=True)
    interjection_csv = f"{save_base_path}ppl.csv"

    if args.test_type not in ["ft_one", "no_ft_one"]:
        raise ValueError("infer_metric.py only supports --test_type ft_one or no_ft_one")

    large_model = LM(args)
    model, tokenizer = large_model.model, large_model.tokenizer
    dataset = DatasetManager(args, tokenizer)
    one_context_df, groundtruth_df = dataset.concat_one_context_infer()

    bertscore = BERTScore(args.language)
    bleuscore = BLEUScore()
    statistic = Statistic(args.language, args.interjection)
    target_phrases = parse_interjections(args.interjection)

    interjection_stats = {phrase: [] for phrase in target_phrases}
    all_hits_ppl = []

    for i in tqdm(range(0, len(one_context_df), args.infer_ratio), desc="Generating"):
        context = one_context_df["Dialogue"][i]
        groundtruth = groundtruth_df["Dialogue"][i]

        generated, phrase_hits = infer_with_phrase_stats(
            context,
            groundtruth,
            model,
            tokenizer,
            target_phrases,
        )

        for hit in phrase_hits:
            phrase = hit["phrase"]
            ppl_t = float(hit["ppl_instant"])
            interjection_stats[phrase].append(ppl_t)
            all_hits_ppl.append(ppl_t)

        generated_clean = strip_ds(generated).lower()
        groundtruth_clean = strip_ds(groundtruth).lower()
        bertscore.compute(groundtruth_clean, generated_clean)
        bleuscore.compute(groundtruth_clean, generated_clean)
        statistic.compute(generated_clean)

    print(f"Backchannel Type - {statistic.types()}")
    print(f"Final Average - {statistic.average()}")
    print(f"Final Scores - BLEU: {bleuscore.average()}")
    print(
        f'BERTScore - P: {bertscore.average("P")}, '
        f'R: {bertscore.average("R")}, '
        f'F1: {bertscore.average("F1")}'
    )

    bertscore.save(save_base_path + "bert.csv")
    bleuscore.save(save_base_path + "bleu.csv")
    statistic.save(save_base_path + "stat.csv")

    lines = ["phrase,mean_instant_ppl\n"]
    for phrase, values in interjection_stats.items():
        if values:
            lines.append(f"{phrase},{sum(values) / len(values):.6f}\n")

    if all_hits_ppl:
        lines.append(f"SUM,{sum(all_hits_ppl) / len(all_hits_ppl):.6f}\n")
    else:
        lines.append("SUM,\n")

    with open(interjection_csv, "w", encoding="utf-8") as file:
        file.writelines(lines)

    print(f"[Interjection PPL] saved to: {interjection_csv}")

"""Non-ablation attribution: does attention localize the culprit, and does it track salience? (GPU).

Referee A: ContextCite and the oracle both work by ablation, so "ContextCite finds the culprit" is
partly two ablation procedures agreeing. A *non-ablation* signal that also localizes the culprit
would break the circularity; one that does not would confirm it. Referee A/C also want the salience
axis validated against an independent signal. Both are answered by one quantity — per-passage
attention mass from the answer tokens back to each passage:

  A1  top-attention == oracle culprit? (culprit accuracy vs ContextCite's ablation-based number)
  A2  does attention mass predict lexical salience? predict causality?  (AUROC of each)

Attention needs eager attention + output_attentions, so we load the HF model directly (4-bit, one
T4). Only k=6 short passages, so the attention tensor fits; cases whose prompt exceeds --max-prompt
tokens are skipped (logged). The first case prints a mapping diagnostic — check it before trusting
the aggregate. Labels (culprit/salient/causal) come from the shipped oracle roles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

import torch
from sklearn.metrics import roc_auc_score

from lineup.config import DEFAULT_MODEL, set_seed
from lineup.data.serialization import read_predictions, read_roles, read_scenarios
from lineup.prompt import build_messages_for


def passage_spans(templated: str, k: int):
    """Char span of each passage [i] in the templated prompt, delimited by the next marker / Question."""
    starts, cur = [], 0
    for i in range(1, k + 1):
        pos = templated.find(f"[{i}]", cur)
        if pos < 0:
            return None
        starts.append(pos)
        cur = pos + 1
    qpos = templated.find("\n\nQuestion:", starts[-1])
    ends = starts[1:] + [qpos if qpos > 0 else len(templated)]
    return list(zip(starts, ends))


@torch.no_grad()
def passage_attention(model, tok, question, chunks, device, max_new_tokens, max_prompt):
    """Per-passage mean attention mass from the answer tokens; None if unmappable / too long."""
    templated = tok.apply_chat_template(build_messages_as_dicts(question, chunks), tokenize=False, add_generation_prompt=True)
    enc = tok(templated, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    if input_ids.shape[1] > max_prompt:
        return None, None
    spans = passage_spans(templated, len(chunks))
    if spans is None:
        return None, None
    offs = enc.offset_mapping[0].tolist()
    tokpass = [next((j for j, (s, e) in enumerate(spans) if s <= a < e), -1) for a, _ in offs]

    gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    answer_ids = gen[0, input_ids.shape[1]:]
    answer = tok.decode(answer_ids, skip_special_tokens=True).strip()

    seq = input_ids.shape[1] + answer_ids.shape[0]
    a0 = input_ids.shape[1]
    out = model(gen[:, :seq], output_attentions=True)
    cols = {j: [t for t in range(a0) if tokpass[t] == j] for j in range(len(chunks))}
    mass = torch.zeros(len(chunks))
    for att in out.attentions:                      # [1, heads, seq, seq] per layer
        recv = att[0, :, a0:seq, :].mean(dim=(0, 1))  # mean over heads + answer rows -> [seq]
        for j, cidx in cols.items():
            if cidx:
                mass[j] += recv[cidx].sum().item()
    mass /= len(out.attentions)
    del out                                          # free the attention tensors immediately
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mass.tolist(), answer


def build_messages_as_dicts(question, chunks):
    return [{"role": m.role, "content": m.content} for m in build_messages_for(question, chunks)]


def process(model, tok, d: Path, device, args, acc):
    stored = {c.qid: c for c in read_roles(d / "roles.jsonl") if not c.original_correct}
    scen = {s.qid: s for s in read_scenarios(d / "scenarios.jsonl")}
    cc = {p.qid: p.predicted_culprit_id for p in read_predictions(d / "predictions.jsonl") if p.method == "contextcite"}
    qids = sorted(q for q in stored if q in scen)
    Random(args.seed).shuffle(qids)
    qids = qids[: args.limit] if args.limit else qids

    wp = att_hit = cc_hit = skipped = 0
    for idx, qid in enumerate(qids):
        sc = scen[qid]
        mass, answer = passage_attention(model, tok, sc.question, sc.chunks, device, args.max_new_tokens, args.max_prompt)
        if mass is None:
            skipped += 1
            continue
        role = {c.chunk_id: c for c in stored[qid].chunk_roles}
        rows = [(mass[j], role[c.chunk_id]) for j, c in enumerate(sc.chunks) if c.chunk_id in role]
        for m, r in rows:
            acc["att"].append(m); acc["sal"].append(int(r.salient)); acc["cau"].append(int(r.causal))
        culprits = [j for j, c in enumerate(sc.chunks) if role.get(c.chunk_id) and role[c.chunk_id].role == "culprit"]
        if culprits:
            wp += 1
            top = max(range(len(sc.chunks)), key=lambda j: mass[j])
            att_hit += int(top in culprits)
            top_cc = next((j for j, c in enumerate(sc.chunks) if c.chunk_id == cc.get(qid)), None)
            cc_hit += int(top_cc in culprits)
        if idx == 0:
            print(f"  [diagnostic case] k={len(sc.chunks)} passages, mass={[round(x,3) for x in mass]}, "
                  f"top-att={max(range(len(mass)), key=lambda j: mass[j])}, culprits={culprits}, answer={answer[:40]!r}")

    cell = d.as_posix().split("results_data/")[-1]
    pct = lambda a, b: 100 * a / b if b else 0.0
    print(f"[{cell}] well-posed n={wp} (skipped {skipped})   "
          f"attention culprit-acc {pct(att_hit, wp):.0f}%   ContextCite {pct(cc_hit, wp):.0f}%")
    acc["wp"] += wp; acc["att_hit"] += att_hit; acc["cc_hit"] += cc_hit


def main() -> None:
    p = argparse.ArgumentParser(description="Attention-based (non-ablation) attribution + salience validation (GPU).")
    p.add_argument("--data-dir", type=Path, nargs="+", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-prompt", type=int, default=1400, help="skip cases whose prompt exceeds this (attention memory on one T4)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map={"": 0}, attn_implementation="eager")
    model.eval()
    device = model.get_input_embeddings().weight.device

    acc = {"att": [], "sal": [], "cau": [], "wp": 0, "att_hit": 0, "cc_hit": 0}
    for d in args.data_dir:
        process(model, tok, d, device, args, acc)

    pct = lambda a, b: 100 * a / b if b else 0.0
    auc = lambda y: roc_auc_score(y, acc["att"]) if len(set(y)) == 2 else float("nan")
    print(f"\n=== POOLED attention attribution ({args.model}) ===\n"
          f"A1  attention culprit-accuracy {pct(acc['att_hit'], acc['wp']):.0f}%   vs ContextCite {pct(acc['cc_hit'], acc['wp']):.0f}%   (well-posed n={acc['wp']})\n"
          f"A2  AUROC(attention -> lexical salience) {auc(acc['sal']):.2f}   AUROC(attention -> causal) {auc(acc['cau']):.2f}   (chunks={len(acc['att'])})\n"
          f"A1: if attention (non-ablation) also localizes >> lexical, the causal signal is real, not ablation-agreement.\n"
          f"A2: higher AUROC for salience than causal => attention tracks resemblance, not effect (validates the split).")


if __name__ == "__main__":
    main()

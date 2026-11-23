"""RAG step for the German medical retrieval evaluation.

The retrieval script writes the top chunks for every question to
results/retrieval_for_rag.json. This script gives those chunks to a language
model and checks how well the answers stay with the source.

Three conditions for the questions that the articles can answer:
  no_context  the model gets only the question
  retrieved   the model gets the top chunks of one retriever (default: Hybrid)
  oracle      the model gets only the best relevant chunk
Questions about topics that are not in the articles are asked in the
"retrieved" condition; the right behaviour there is to say so.

Automatic checks per answer (see the README for what they can and cannot see):
  correct          every key-term group of the question appears in the answer
  abstained        the answer says that the texts do not contain it
  supported        share of answer sentences whose content words appear in the
                   given chunks (a lexical check, not a proof)
  valid citation   the answer cites [n] and n is one of the given chunks

Run:
  python german_medical_retrieval_eval.py
  python rag_eval.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct
  python rag_eval.py --backend openai --model MODEL --base-url http://localhost:11434/v1
"""
import argparse
import json
import os
import re
import time

import pandas as pd

ABSTAIN = "Das steht nicht in den Texten."
SYSTEM_RAG = (
    "Du beantwortest medizinische Fragen ausschließlich anhand der nummerierten Textauszüge. "
    "Antworte kurz auf Deutsch in höchstens drei Sätzen und nenne die Nummer des Auszugs in "
    f"eckigen Klammern, zum Beispiel [2]. Wenn die Antwort nicht in den Auszügen steht, antworte genau: {ABSTAIN}"
)
SYSTEM_PLAIN = "Du beantwortest medizinische Fragen kurz auf Deutsch in höchstens drei Sätzen."
STOP = set(
    "der die das den dem des ein eine einer einem einen eines und oder aber auch nicht nur wird werden "
    "wurde wurden ist sind war waren hat haben kann können als bei mit von für auf aus nach über unter "
    "zum zur im in am an es sich dass wenn wie was wer wo wann welche welcher welches diese dieser "
    "dieses sowie durch bis ohne mehr sehr etwa laut text auszug auszügen auszuges damit dabei dazu "
    "dafür dann denn doch noch schon sein seine ihre ihrer ihren außerdem zudem hierbei zusätzlich insgesamt".split()
)


def build_messages(question, ctx):
    if not ctx:
        return [{"role": "system", "content": SYSTEM_PLAIN}, {"role": "user", "content": f"Frage: {question}"}]
    block = "\n\n".join(f"[{i}] ({c['doc']}, {c['section'] or 'Einleitung'})\n{c['text']}" for i, c in enumerate(ctx, 1))
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Textauszüge:\n\n{block}\n\nFrage: {question}"}]


def make_generator(args):
    if args.backend == "dummy":  # only to test the pipeline, the answers are meaningless
        def gen(messages):
            m = re.search(r"\[1\][^\n]*\n(.+?)(\n\n|$)", messages[-1]["content"], re.S)
            return f"{' '.join(m.group(1).split()[:25])} [1]" if m else "Keine Angabe."
        return gen
    if args.backend == "openai":  # any OpenAI-compatible server; the key is read from OPENAI_API_KEY
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

        def gen(messages):
            r = client.chat.completions.create(model=args.model, messages=messages,
                                               temperature=0, max_tokens=args.max_new_tokens)
            return (r.choices[0].message.content or "").strip()
        return gen
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    def gen(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return gen


def make_jobs(data, retriever, conditions, limit):
    jobs = []
    for qi, item in enumerate(data["questions"][:limit] if limit else data["questions"]):
        for cond in conditions:
            ctx = {"no_context": [], "oracle": [item["oracle"]], "retrieved": item["retrieved"][retriever]}[cond]
            jobs.append({"kind": "answerable", "cond": cond, "idx": qi, "item": item, "ctx": ctx})
    if "retrieved" in conditions:
        for ui, item in enumerate(data["unanswerable"]):
            jobs.append({"kind": "unanswerable", "cond": "retrieved", "idx": ui, "item": item,
                         "ctx": item["retrieved"][retriever]})
    return jobs


def is_abstain(answer):
    return "nicht in den texten" in answer.lower()[:90]


def content_words(sentence):
    """Nouns, names, abbreviations and numbers: capitalised words and words with digits (German nouns are capitalised)."""
    sentence = re.sub(r"\[\d+\]", " ", sentence)
    words = re.findall(r"[A-Za-zÄÖÜäöüß0-9][A-Za-zÄÖÜäöüß0-9\-]*", sentence)
    keep = []
    for w in words:
        lw = w.lower()
        if lw in STOP:
            continue
        if w.isdigit() or (any(ch.isdigit() for ch in w) and len(w) >= 3) or (w[0].isupper() and len(w) >= 3):
            keep.append(lw)
    return keep


def support(answer, ctx):
    """Share of sentences in which at least 80 % of the content words occur in the chunks, and the words that do not.

    Plain numbers must occur as whole numbers, other words as part of a word in the chunks."""
    text = " ".join(c["text"] for c in ctx).lower()
    numbers = set(re.findall(r"\d+", text))
    scored, ok, missing = 0, 0, []
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        words = content_words(sent)
        if not words:
            continue
        miss = [w for w in words
                if not ((w in numbers) if w.isdigit() else (w in text or (len(w) >= 7 and w[:-2] in text)))]
        scored += 1
        ok += (len(words) - len(miss)) / len(words) >= 0.8
        missing += miss
    return (ok / scored if scored else None), missing


def score(job, answer):
    item, ctx = job["item"], job["ctx"]
    row = {"condition": job["cond"], "kind": job["kind"], "question": item["q"],
           "lang": item.get("lang", "de"), "answer": answer, "abstained": is_abstain(answer)}
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    row["retrieval_hit"] = any(c["relevant"] for c in ctx) if ctx else None
    row["citation_valid"] = (bool(cites) and all(1 <= c <= len(ctx) for c in cites)) if ctx else None
    row["cited_relevant"] = (any(1 <= c <= len(ctx) and ctx[c - 1]["relevant"] for c in cites)) if ctx else None
    if job["kind"] == "answerable":
        low = answer.lower()
        row["correct"] = all(any(t.lower() in low for t in group) for group in item["terms"])
    else:
        row["correct"] = None
    share, missing = (None, []) if (not ctx or row["abstained"]) else support(answer, ctx)
    row["supported_share"] = share
    row["fully_supported"] = None if share is None else share == 1.0
    row["unsupported_words"] = " ".join(sorted(set(missing)))
    row["manual_label"] = ""
    return row


def frac(series):
    s = series.dropna()
    return f"{int(s.sum())}/{len(s)} ({100 * s.mean():.0f} %)" if len(s) else "n/a"


def summarize(df, meta):
    ans = df[df.kind == "answerable"]
    lines = [f"Model: {meta['model']} | retriever: {meta['retriever']} | top {meta['top_k']} chunks | "
             f"{ans.question.nunique()} answerable and {df[df.kind == 'unanswerable'].question.nunique()} unanswerable questions\n",
             "Questions that the articles can answer\n",
             "| Condition | Correct | Abstained | Fully supported | Correct and fully supported | Valid citation |",
             "|---|---|---|---|---|---|"]
    for cond in ["no_context", "retrieved", "oracle"]:
        d = ans[ans.condition == cond]
        if d.empty:
            continue
        both = (d.correct & (d.fully_supported == True)).where(d.fully_supported.notna())
        lines.append(f"| {cond} | {frac(d.correct)} | {frac(d.abstained)} | {frac(d.fully_supported)} | "
                     f"{frac(both)} | {frac(d.citation_valid)} |")
    d = ans[ans.condition == "retrieved"]
    if not d.empty:
        lines += ["", "Retrieved condition, split by whether the top chunks contained a relevant chunk\n",
                  "| Top chunks | Questions | Correct | Cited a relevant chunk |", "|---|---|---|---|"]
        for hit, label in [(True, "relevant chunk present"), (False, "no relevant chunk")]:
            s = d[d.retrieval_hit == hit]
            if len(s):
                lines.append(f"| {label} | {len(s)} | {frac(s.correct)} | {frac(s.cited_relevant)} |")
    un = df[df.kind == "unanswerable"]
    if not un.empty:
        lines += ["", "Questions about topics that are not in the articles\n",
                  f"Said that the texts do not contain it: {frac(un.abstained)}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai", "dummy"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--retriever", default="Hybrid", choices=["BM25", "Dense", "Hybrid"])
    ap.add_argument("--conditions", default="no_context,retrieved,oracle")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0, help="only the first N answerable questions")
    ap.add_argument("--input", default="results/retrieval_for_rag.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    conditions = args.conditions.split(",")
    jobs = make_jobs(data, args.retriever, conditions, args.limit)
    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, "rag_generations.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            rec = json.loads(line)
            cache[rec["key"]] = rec["answer"]
    gen = make_generator(args) if any(
        f"{args.model}|{args.retriever}|{j['kind']}|{j['cond']}|{j['idx']}" not in cache for j in jobs) else None

    rows, t0 = [], time.time()
    with open(cache_path, "a", encoding="utf-8") as cache_file:
        for n, job in enumerate(jobs, 1):
            key = f"{args.model}|{args.retriever}|{job['kind']}|{job['cond']}|{job['idx']}"
            if key not in cache:
                cache[key] = gen(build_messages(job["item"]["q"], job["ctx"]))
                cache_file.write(json.dumps({"key": key, "answer": cache[key]}, ensure_ascii=False) + "\n")
                cache_file.flush()
                print(f"[{n}/{len(jobs)}] {job['cond']:10s} {time.time() - t0:6.0f}s  {job['item']['q'][:60]}")
            rows.append(score(job, cache[key]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "rag_answers.csv"), index=False)
    text = summarize(df, {"model": args.model, "retriever": args.retriever, "top_k": data["top_k"]})
    open(os.path.join(args.out, "rag_summary.md"), "w", encoding="utf-8").write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()

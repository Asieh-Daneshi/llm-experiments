# # Retrieval evaluation on German cardiology texts
# Question: for German medical text, does hybrid retrieval (BM25 + embeddings) find the right passage more often than BM25 or embeddings alone?
# Data: German Wikipedia articles on cardiology and kidney topics. Test set: hand-written questions with the source article and key terms that a relevant passage must contain.
# Metrics: Hit@k (is a relevant passage in the top k) and MRR (mean of 1 / rank of the first relevant passage). Comparison: paired bootstrap confidence interval over questions.

# %%
# ## 0. Setup
# `pip install rank_bm25 sentence-transformers requests pandas`

# %%
import json, re, random
import numpy as np
import pandas as pd
import requests
from rank_bm25 import BM25Okapi

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# %%
# ## 1. Download the articles
# Plain text from the German Wikipedia API. `redirects=1` follows renamed titles.
# %%
ARTICLES = [
    "Herzinsuffizienz", "Vorhofflimmern", "Herzinfarkt", "Akutes Nierenversagen",
    "Koronare Herzkrankheit", "Koronararterien-Bypass", "Herz-Lungen-Maschine",
    "Aortenklappenstenose", "Arterielle Hypertonie", "Lungenembolie",
    "Endokarditis", "Perikardtamponade", "TAVI",
]

def fetch_article(title):
    r = requests.get(
        "https://de.wikipedia.org/w/api.php",
        params={"action": "query", "format": "json", "prop": "extracts",
                "explaintext": 1, "redirects": 1, "titles": title},
        headers={"User-Agent": "retrieval-eval-study/0.1 (asieh.daneshi@gmail.com)"},
        timeout=30,
    )
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("title", title), page.get("extract", "")

docs = {}
for t in ARTICLES:
    real_title, text = fetch_article(t)
    docs[t] = text
    print(f"{t:28s} -> {real_title:30s} {len(text):7d} chars")

json.dump(docs, open("articles.json", "w", encoding="utf-8"), ensure_ascii=False)

# %%
# ## 2. Chunking
# Split at section headings (`== ... ==`) and paragraphs, then merge paragraphs until a chunk reaches about `max_words`.

# %%
SKIP_SECTIONS = {"Literatur", "Weblinks", "Einzelnachweise", "Siehe auch", "Quellen", "Anmerkungen"}
MIN_WORDS = 20

def chunk_article(title, text, max_words=150):
    text = re.sub(r"\n=+ (.+?) =+\n", r"\n\n### \1\n\n", "\n" + text)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur, section = [], [], ""
    def flush():
        if cur and section not in SKIP_SECTIONS:
            chunks.append({"doc": title, "section": section, "text": " ".join(cur)})
    for p in paras:
        if p.startswith("### "):
            flush(); cur = []
            section = p[4:].strip()
            continue
        words = p.split()
        # split very long paragraphs into pieces of max_words
        for start in range(0, len(words), max_words):
            cur.append(" ".join(words[start:start + max_words]))
            if sum(len(x.split()) for x in cur) >= max_words:
                flush(); cur = []
    flush()
    return [c for c in chunks if len(c["text"].split()) >= MIN_WORDS]

def build_chunks(docs, max_words=150):
    out = []
    for title, text in docs.items():
        out += chunk_article(title, text, max_words)
    for i, c in enumerate(out):
        c["id"] = i
    return out

chunks = build_chunks(docs, max_words=150)
print(len(chunks), "chunks")
print(pd.Series([len(c["text"].split()) for c in chunks]).describe())

# %%
# ## 3. Test set
# Each question has the article that should answer it (`doc`) and key-term groups (`terms`). A chunk counts as relevant if it comes from that article and contains at least one term from every group (case-insensitive, substring match).
#
# The last 4 questions are in English, to test cross-language retrieval.

# %%
TESTSET = [
 {"q": "Wie wird der Schweregrad einer Herzinsuffizienz eingeteilt?", "doc": "Herzinsuffizienz", "terms": [["NYHA"]]},
 {"q": "Welche Medikamente werden bei Herzinsuffizienz eingesetzt?", "doc": "Herzinsuffizienz", "terms": [["ACE-Hemmer"], ["Betablocker", "Beta-Blocker"]]},
 {"q": "Welcher Laborwert hilft bei der Diagnose einer Herzinsuffizienz?", "doc": "Herzinsuffizienz", "terms": [["BNP"]]},
 {"q": "Was ist eine Herzschwäche?", "doc": "Herzinsuffizienz", "terms": [["Herzschwäche", "Herzmuskelschwäche"]]},
 {"q": "Wie wird das Schlaganfallrisiko bei Vorhofflimmern abgeschätzt?", "doc": "Vorhofflimmern", "terms": [["CHA"]]},
 {"q": "Welche Gerinnungshemmer bekommen Patienten mit Vorhofflimmern?", "doc": "Vorhofflimmern", "terms": [["antikoagul"], ["Vitamin-K", "Phenprocoumon", "Marcumar", "NOAK", "DOAK", "Apixaban", "Rivaroxaban", "Dabigatran", "Edoxaban"]]},
 {"q": "Was ist eine Kardioversion?", "doc": "Vorhofflimmern", "terms": [["Kardioversion"]]},
 {"q": "Welcher Laborwert zeigt einen Herzinfarkt an?", "doc": "Herzinfarkt", "terms": [["Troponin"]]},
 {"q": "Was ist der Unterschied zwischen STEMI und NSTEMI?", "doc": "Herzinfarkt", "terms": [["NSTEMI"]]},
 {"q": "Wie wird ein Herzinfarkt im Herzkatheterlabor behandelt?", "doc": "Herzinfarkt", "terms": [["PCI", "perkutan", "Stent", "Ballon"]]},
 {"q": "Nach welchen Kriterien wird ein akutes Nierenversagen eingeteilt?", "doc": "Akutes Nierenversagen", "terms": [["KDIGO", "RIFLE", "AKIN"]]},
 {"q": "Welcher Laborwert steigt beim akuten Nierenversagen an?", "doc": "Akutes Nierenversagen", "terms": [["Kreatinin"]]},
 {"q": "Wann braucht man beim akuten Nierenversagen eine Dialyse?", "doc": "Akutes Nierenversagen", "terms": [["Dialyse", "Nierenersatz"]]},
 {"q": "Was ist Angina pectoris?", "doc": "Koronare Herzkrankheit", "terms": [["Angina pectoris"]]},
 {"q": "Welche Risikofaktoren hat die koronare Herzkrankheit?", "doc": "Koronare Herzkrankheit", "terms": [["Rauch"], ["Diabetes"]]},
 {"q": "Mit welcher Untersuchung werden verengte Herzkranzgefäße sicher nachgewiesen?", "doc": "Koronare Herzkrankheit", "terms": [["Koronarangiographie", "Koronarangiografie", "Herzkatheter"]]},
 {"q": "Welche Gefäße werden als Bypass verwendet?", "doc": "Koronararterien-Bypass", "terms": [["thoracica", "Brustwandarterie", "Mammaria", "saphena", "Beinvene"]]},
 {"q": "Wozu dient eine Herz-Lungen-Maschine?", "doc": "Herz-Lungen-Maschine", "terms": [["Kreislauf", "Blut"]]},
 {"q": "Was ist TAVI?", "doc": "TAVI", "terms": [["TAVI", "Transkatheter"]]},
 {"q": "Welche Beschwerden macht eine Aortenklappenstenose?", "doc": "Aortenklappenstenose", "terms": [["Synkope", "Schwindel", "Bewusstlos"]]},
 {"q": "Ab welchem Blutdruck spricht man von Bluthochdruck?", "doc": "Arterielle Hypertonie", "terms": [["140"], ["90"]]},
 {"q": "Welche Organschäden kann Bluthochdruck verursachen?", "doc": "Arterielle Hypertonie", "terms": [["Schlaganfall", "Niere", "Herzinsuffizienz"]]},
 {"q": "Welcher Bluttest hilft bei Verdacht auf Lungenembolie?", "doc": "Lungenembolie", "terms": [["D-Dimer"]]},
 {"q": "Woher stammt das Gerinnsel bei einer Lungenembolie meistens?", "doc": "Lungenembolie", "terms": [["Thrombose"]]},
 {"q": "Nach welchen Kriterien wird eine Endokarditis diagnostiziert?", "doc": "Endokarditis", "terms": [["Duke"]]},
 {"q": "Welche Bakterien verursachen eine Endokarditis?", "doc": "Endokarditis", "terms": [["Staphylo", "Streptok"]]},
 {"q": "Was passiert bei einer Herzbeuteltamponade?", "doc": "Perikardtamponade", "terms": [["Perikard", "Herzbeutel"]]},
 {"q": "Which blood marker indicates a heart attack?", "doc": "Herzinfarkt", "terms": [["Troponin"]], "lang": "en"},
 {"q": "How is stroke risk estimated in atrial fibrillation?", "doc": "Vorhofflimmern", "terms": [["CHA"]], "lang": "en"},
 {"q": "Which blood test is used when pulmonary embolism is suspected?", "doc": "Lungenembolie", "terms": [["D-Dimer"]], "lang": "en"},
 {"q": "What is TAVI used for?", "doc": "TAVI", "terms": [["TAVI", "Transkatheter"]], "lang": "en"},
]
for t in TESTSET:
    t.setdefault("lang", "de")

def is_relevant(chunk, item):
    if chunk["doc"] != item["doc"]:
        return False
    text = chunk["text"].lower()
    return all(any(term.lower() in text for term in group) for group in item["terms"])

def relevant_ids(item, chunks):
    return {c["id"] for c in chunks if is_relevant(c, item)}

# Sanity check: every question needs at least one relevant chunk.
for t in TESTSET:
    n = len(relevant_ids(t, chunks))
    flag = "   <-- no relevant chunk" if n == 0 else ""
    print(f"{n:3d}  {t['q']}{flag}")

# %%
# ## 4. Three retrievers
# - **BM25**: keyword search (classic information retrieval).
# - **Dense**: embeddings with `intfloat/multilingual-e5-small` (needs the prefixes `query:` and `passage:`), cosine similarity.
# - **Hybrid**: reciprocal rank fusion (RRF) of both rankings: score(d) = Σ 1 / (60 + rank(d)).

# %%
def tokenize(s):
    return re.findall(r"\w+", s.lower())

class BM25Retriever:
    def __init__(self, chunks):
        self.ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    def rank(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        return [self.ids[i] for i in order]

class DenseRetriever:
    def __init__(self, chunks, model_name="intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.ids = [c["id"] for c in chunks]
        self.emb = self.model.encode(["passage: " + c["text"] for c in chunks],
                                     normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    def rank(self, query):
        q = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        order = np.argsort(-(self.emb @ q))
        return [self.ids[i] for i in order]

class HybridRetriever:
    def __init__(self, retrievers, k=60):
        self.retrievers, self.k = retrievers, k
    def rank(self, query):
        score = {}
        for r in self.retrievers:
            for pos, cid in enumerate(r.rank(query), start=1):
                score[cid] = score.get(cid, 0.0) + 1.0 / (self.k + pos)
        return sorted(score, key=score.get, reverse=True)

bm25 = BM25Retriever(chunks)
dense = DenseRetriever(chunks)
hybrid = HybridRetriever([bm25, dense])
RETRIEVERS = {"BM25": bm25, "Dense": dense, "Hybrid": hybrid}

# %%
# ## 5. Evaluation
# Per question: rank of the first relevant chunk. From that: Hit@1, Hit@5, Hit@10 and reciprocal rank (RR). MRR is the mean RR.

# %%
def first_relevant_rank(ranking, rel):
    for pos, cid in enumerate(ranking, start=1):
        if cid in rel:
            return pos
    return None

rows = []
for qi, item in enumerate(TESTSET):
    rel = relevant_ids(item, chunks)
    for name, r in RETRIEVERS.items():
        rank = first_relevant_rank(r.rank(item["q"]), rel)
        rows.append({"qi": qi, "lang": item["lang"], "retriever": name, "rank": rank,
                     "hit1": rank is not None and rank <= 1,
                     "hit5": rank is not None and rank <= 5,
                     "hit10": rank is not None and rank <= 10,
                     "rr": 0.0 if rank is None else 1.0 / rank})
res = pd.DataFrame(rows)

summary = res.groupby("retriever")[["hit1", "hit5", "hit10", "rr"]].mean().rename(columns={"rr": "MRR"})
print(summary.round(3))
print()
print(res.groupby(["lang", "retriever"])[["hit5", "rr"]].mean().round(3))

# %%
# ## 6. Is the difference real? Paired bootstrap
# Resample questions with replacement (same questions for both retrievers), compute the difference in MRR each time, and take the 2.5% and 97.5% percentiles as a 95% confidence interval. With about 30 questions the intervals are wide.

# %%
def paired_bootstrap(res, a, b, metric="rr", n_boot=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    A = res[res.retriever == a].sort_values("qi")[metric].to_numpy(dtype=float)
    B = res[res.retriever == b].sort_values("qi")[metric].to_numpy(dtype=float)
    d = B - A
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

for a, b in [("BM25", "Hybrid"), ("Dense", "Hybrid"), ("BM25", "Dense")]:
    m, lo, hi = paired_bootstrap(res, a, b)
    print(f"MRR {b} - {a}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

# %%
# ## 7. Error analysis
# List every question where a retriever has no relevant chunk in the top 5, with the ranks and the top chunk of the first retriever that missed.

# %%
for qi, item in enumerate(TESTSET):
    r = res[res.qi == qi].set_index("retriever")["rank"]
    misses = [n for n in RETRIEVERS if not (pd.notna(r[n]) and r[n] <= 5)]
    if misses:
        print(f"[{item['lang']}] {item['q']}")
        print("     ranks:", {n: (None if pd.isna(r[n]) else int(r[n])) for n in RETRIEVERS}, " misses:", misses)
        top = RETRIEVERS[misses[0]].rank(item["q"])[0]
        print("     top chunk of", misses[0], "->", chunks[top]["doc"], "|", chunks[top]["text"][:160], "...")
        print()

# %%
# ## 8. Chunk size
# To compare chunk sizes, change `max_words` in `build_chunks` (for example 80 or 300) and re-run sections 2 to 6.

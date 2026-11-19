# German medical retrieval evaluation

Question: for German medical text, does hybrid retrieval (BM25 + embeddings, combined with reciprocal rank fusion) find the right passage more often than BM25 or embeddings alone?

## What the script does

- Downloads 13 German Wikipedia articles on cardiology and kidney topics through the Wikipedia API and splits them into chunks of about 150 words (section headings and paragraphs are respected).
- Uses a hand-written test set of 30 questions: 26 in German and 4 in English to test cross-language retrieval. Each question names the article that should answer it and key terms that a relevant chunk must contain.
- Compares three retrievers: BM25 (keyword search), dense retrieval with `intfloat/multilingual-e5-small`, and a hybrid of both.
- Reports Hit@1, Hit@5, Hit@10 and MRR, a paired bootstrap 95 % confidence interval for the MRR differences, and a list of the questions that each retriever misses in the top 5.

The data are public encyclopedia texts, not clinical data. With about 30 questions the confidence intervals are wide, so differences between retrievers should be read with care.

## Run

```bash
pip install -r requirements.txt
python german_medical_retrieval_eval.py
```

The script needs internet access (Wikipedia and the model download) and writes `articles.json` next to it. It is organised in `# %%` cells, so it can also be run cell by cell in an editor.

To compare chunk sizes, change `max_words` in `build_chunks` and run it again.

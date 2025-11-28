import json
import numpy as np
import os

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_PATH = 'Data/docs.jsonl'
INDEX_PATH = 'data/index.npy'
META_PATH = 'data/meta.jsonl'


def load_docs():
    docs = []
    with open(DATA_PATH,"r",encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def build_index():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    docs = load_docs()
    all_sentences = []

    meta = []

    for doc in docs:
        doc_id = doc['doc_id']
        for i,sent in enumerate(doc["sentences"]):
            all_sentences.append(sent)
            meta.append({
                "doc_id":doc_id,
                "sent_index":i
            })
            embeddings = model.encode(all_sentences,batch_size=32,show_progress_bar=True)
            embeddings = np.array(embeddings,dtype=np.float32)

        os.makedirs(os.path.dirname(INDEX_PATH),exist_ok=True)
        np.save(INDEX_PATH,embeddings)

        with open(META_PATH,'w',encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m)+"\n")

if __name__=="__main__":
    build_index()

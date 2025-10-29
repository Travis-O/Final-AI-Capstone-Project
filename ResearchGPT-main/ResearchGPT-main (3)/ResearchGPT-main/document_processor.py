import os
import json
from typing import List, Tuple
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentProcessor:
    def __init__(self, data_dir: str = None, chunk_size: int = 800, overlap: int = 200):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents = {}
        self.chunks = []
        self.vectorizer = None
        self.vectors = None

    def _read_pdf(self, path: str) -> str:
        text_parts = []
        try:
            with open(path, 'rb') as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    text_parts.append(page_text)
        except Exception:
            return ""
        return "\n".join(text_parts)

    def _read_txt(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return fh.read()
        except Exception:
            with open(path, 'r', encoding='latin-1') as fh:
                return fh.read()

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def process_document(self, file_path: str) -> List[Tuple[str, str]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        _, ext = os.path.splitext(file_path.lower())
        raw = self._read_pdf(file_path) if ext == ".pdf" else self._read_txt(file_path)
        text = self._clean_text(raw)

        doc_id = os.path.basename(file_path)
        self.documents[doc_id] = {"path": file_path, "length": len(text)}

        chunks, start, L = [], 0, len(text)
        if L == 0:
            return []
        while start < L:
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, doc_id))
            if end >= L:
                break
            start = end - self.overlap

        self.chunks.extend(chunks)
        return chunks

    def build_search_index(self, force_rebuild: bool = False):
        if not self.chunks:
            if os.path.isdir(self.data_dir):
                for fname in os.listdir(self.data_dir):
                    if fname.lower().endswith((".pdf", ".txt")):
                        p = os.path.join(self.data_dir, fname)
                        try:
                            self.process_document(p)
                        except Exception:
                            continue
        texts = [c[0] for c in self.chunks]
        if not texts:
            self.vectorizer = None
            self.vectors = None
            return
        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
        self.vectors = self.vectorizer.fit_transform(texts)

    def find_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        if self.vectorizer is None or self.vectors is None:
            self.build_search_index()
        if self.vectorizer is None or self.vectors is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.vectors).flatten()
        if len(sims) == 0:
            return []
        idx_sorted = np.argsort(-sims)[:top_k]
        out = []
        for idx in idx_sorted:
            score = float(sims[idx])
            chunk_text, doc_id = self.chunks[idx]
            out.append((chunk_text, score, doc_id))
        return out

    def get_document_stats(self):
        return {
            "num_docs": len(self.documents),
            "num_chunks": len(self.chunks),
            "documents": self.documents
        }

    def get_all_chunks(self):
        return list(self.chunks)

    def save_index(self, path: str):
        meta = {"documents": self.documents, "chunks": self.chunks}
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        if self.vectorizer is not None:
            import pickle
            with open(path + ".vectorizer.pkl", "wb") as fh:
                pickle.dump(self.vectorizer, fh)

    def load_index(self, path: str):
        if os.path.exists(path + ".meta.json"):
            with open(path + ".meta.json", "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.documents = meta.get("documents", {})
            self.chunks = [tuple(x) for x in meta.get("chunks", [])]
        if os.path.exists(path + ".vectorizer.pkl"):
            import pickle
            with open(path + ".vectorizer.pkl", "rb") as fh:
                self.vectorizer = pickle.load(fh)
            texts = [c[0] for c in self.chunks]
            if texts:
                self.vectors = self.vectorizer.transform(texts)

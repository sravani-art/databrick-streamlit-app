# app/vectorstore.py
import faiss
import numpy as np
import threading
import time
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# VERSION REGISTRY
# Tracks the current active version for each process_id.
# Stored separately from the vector store so version metadata survives
# a clear_process() call and is cheap to query without touching FAISS.
#
# Structure:
#   _version_registry[process_id] = {
#       "current_version": int,       # latest version number (1-based)
#       "history": [                  # one entry per version, oldest first
#           {
#               "version": int,
#               "uploaded_at": int,   # unix timestamp
#               "filename": str,
#               "chunks": int,
#           },
#           ...
#       ]
#   }
# ─────────────────────────────────────────────────────────────────────────────

_version_registry: Dict[str, Dict] = {}
_version_lock = threading.Lock()


def register_version(process_id: str, filename: str, chunks: int) -> int:
    """
    Increment the version counter for a process and record the upload event.
    Returns the new version number.
    """
    with _version_lock:
        entry = _version_registry.setdefault(process_id, {"current_version": 0, "history": []})
        entry["current_version"] += 1
        version = entry["current_version"]
        entry["history"].append({
            "version":     version,
            "uploaded_at": int(time.time()),
            "filename":    filename,
            "chunks":      chunks,
        })
        return version


def get_version_info(process_id: str) -> Optional[Dict]:
    """Return the full version registry entry for a process_id, or None if unknown."""
    return _version_registry.get(process_id)


def get_current_version(process_id: str) -> int:
    """Return the current version number (0 if no uploads recorded yet)."""
    entry = _version_registry.get(process_id)
    return entry["current_version"] if entry else 0

class InMemoryFaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.lock = threading.Lock()
        self.metadatas: List[Dict[str, Any]] = []
        # Per-process cache: process_id -> faiss.IndexFlatIP built from that process's vectors
        # Invalidated on add() or clear_process() for the affected process_id.
        self._process_index_cache: Dict[str, Any] = {}
        # O(1) lookup: process_id -> list of metadata dicts for that process
        self._meta_by_process: Dict[str, List[Dict[str, Any]]] = {}

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        embeddings: (N, dim) numpy float32
        metadatas: list length N
        """
        # if embeddings.ndim == 1:
        #     embeddings = embeddings.reshape(1, -1)
        # faiss.normalize_L2(embeddings)
        # with self.lock:
        #     self.index.add(embeddings)
        #     self.metadatas.extend(metadatas)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        assert embeddings.shape[0] == len(metadatas), "Embeddings / metadata length mismatch"
        for i, meta in enumerate(metadatas):
            meta["embedding"] = embeddings[i]
        with self.lock:
            self.index.add(embeddings)
            self.metadatas.extend(metadatas)
            # Update O(1) lookup dict
            for meta in metadatas:
                pid = meta.get("process_id")
                if pid is not None:
                    self._meta_by_process.setdefault(pid, []).append(meta)
            # Invalidate per-process index cache for affected process_ids
            affected = {m.get("process_id") for m in metadatas if m.get("process_id")}
            for pid in affected:
                self._process_index_cache.pop(pid, None)

    def query(self, embedding: np.ndarray, top_k: int = 5):
        """Global query across all processes"""
        embedding = embedding.astype('float32')
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        with self.lock:
            if self.index.ntotal == 0:
                return []
            D, I = self.index.search(embedding, top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                meta = self.metadatas[idx]
                results.append({"score": float(score), "metadata": meta})
            return results

    def query_by_process(self, process_id: str, embedding: np.ndarray, top_k: int = 5):
        """
        Restrict query to vectors belonging to the given process_id.
        Uses a cached per-process FAISS index — rebuilt only when vectors
        for that process change (add / clear_process), not on every query.
        """
        embedding = embedding.astype('float32')
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)

        with self.lock:
            if self.index.ntotal == 0:
                return []

            # Build and cache the per-process index if not already present
            if process_id not in self._process_index_cache:
                process_metas = self._meta_by_process.get(process_id, [])
                if not process_metas:
                    return []
                sub_vectors = [m["embedding"] for m in process_metas if "embedding" in m]
                if not sub_vectors:
                    return []
                sub_array = np.vstack(sub_vectors).astype("float32")
                faiss.normalize_L2(sub_array)
                sub_index = faiss.IndexFlatIP(self.dim)
                sub_index.add(sub_array)
                self._process_index_cache[process_id] = (sub_index, process_metas)

            sub_index, process_metas = self._process_index_cache[process_id]
            D, I = sub_index.search(embedding, min(top_k, sub_index.ntotal))

            results = []
            for score, sid in zip(D[0], I[0]):
                if sid < 0:
                    continue
                results.append({"score": float(score), "metadata": process_metas[sid]})
            return results
            
    # def query_by_process(self, process_id: str, embedding: np.ndarray, top_k: int = 5):
    #     """
    #     Restrict query to vectors belonging to the given process_id.
    #     Creates a temporary FAISS index for those entries only.
    #     """
    #     embedding = embedding.astype('float32')
    #     if embedding.ndim == 1:
    #         embedding = embedding.reshape(1, -1)
    #     faiss.normalize_L2(embedding)

    #     with self.lock:
    #         if self.index.ntotal == 0:
    #             return []

    #         # if not self.metadatas:
    #         #     return []

    #         # indices = [i for i, m in enumerate(self.metadatas) if m.get("process_id") == process_id]
    #         indices = [(i, m) for i, m in enumerate(self.metadatas) if m.get("process_id") == process_id]
    #         if not indices:
    #             return []

    #         sub_index = faiss.IndexFlatIP(self.dim)         # Building a temp index for only process_id embeddings
    #         # vectors = []
    #         # for i in indices:
    #         #     vec = self.index.reconstruct(i)
    #         #     vectors.append(vec)
    #         # sub_vectors = np.array(vectors, dtype='float32')

    #         sub_vectors = [m["embedding"] for _, m in indices if "embedding" in m]
    #         if not sub_vectors:
    #             return []

    #         sub_vectors = np.vstack(sub_vectors).astype("float32")
            
    #         faiss.normalize_L2(sub_vectors)
    #         sub_index.add(sub_vectors)

    #         D, I = sub_index.search(embedding, min(top_k, len(sub_vectors)))

    #         results = []
    #         for score, sid in zip(D[0], I[0]):
    #             # if sid < 0:
    #             #     continue
    #             # meta = self.metadatas[indices[sid]]
    #             meta = indices[sid][1]
    #             results.append({"score": float(score), "metadata": meta})
    #         return results

    def clear_process(self, process_id: str):
        """
        Removes all vectors for a given process_id (not super efficient; rebuild index).
        Use when process maps change and you want to replace vectors for that process.
        """
        # with self.lock:
        #     keep = [m for m in self.metadatas if m.get("process_id") != process_id]
        #     if len(keep) == len(self.metadatas):
        #         return
        #     new_index = faiss.IndexFlatIP(self.dim)
        #     self.index = new_index
        #     self.metadatas = keep
        with self.lock:
            keep_meta = [m for m in self.metadatas if m.get("process_id") != process_id]
            if len(keep_meta) == len(self.metadatas):
                return
            new_index = faiss.IndexFlatIP(self.dim)
            emb_list = [m["embedding"] for m in keep_meta if "embedding" in m]
            if emb_list:
                emb_array = np.vstack(emb_list).astype("float32")
                faiss.normalize_L2(emb_array)
                new_index.add(emb_array)

            self.index = new_index
            self.metadatas = keep_meta
            # Rebuild O(1) lookup without the cleared process
            self._meta_by_process.pop(process_id, None)
            # Invalidate cache for cleared process
            self._process_index_cache.pop(process_id, None)

    def snapshot(self, path_prefix: str):
        """Optional snapshot of metadata + index to disk (not used by default)."""
        with self.lock:
            faiss.write_index(self.index, f"{path_prefix}.index")
            import json
            with open(f"{path_prefix}.meta.json", "w", encoding="utf-8") as f:
                json.dump(self.metadatas, f)

    def get_metadata_by_process(self, process_id: str) -> list:
        """Retrieve all metadata entries for a given process_id."""
        return [m for m in self.metadatas if m.get("process_id") == process_id]

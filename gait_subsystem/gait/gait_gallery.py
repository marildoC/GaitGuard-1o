"""
gait/gait_gallery.py - FAISS GALLERY & MATCHING LOGIC
"""
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import logging
import pickle
from dataclasses import dataclass
from gait_subsystem.gait.config import GaitConfig
import faiss

logger = logging.getLogger(__name__)

@dataclass
class GaitIdentityData:
    """
    Data structure representing a unique identity in the gallery.
    
    Attributes:
        identity_id: The unique string ID (e.g., "Francesco").
        faiss_id: The integer ID used internally by FAISS.
        ema_embedding: The representative embedding vector (averaged over time).
        raw_embeddings: List of recent raw embeddings (optional history).
        num_updates: How many times this identity has been updated/seen.
        category: User category (e.g., "resident", "guest").
    """
    identity_id: str
    faiss_id: int
    ema_embedding: np.ndarray       
    raw_embeddings: List[np.ndarray] 
    num_updates: int = 0
    category: str = "resident"
    anthro_mean: Optional[np.ndarray] = None

@dataclass
class PersonSummary:
    """DTO for listing persons via CLI."""
    person_id: str
    name: str
    category: str
    num_templates: int

class FaissIndexWrapper:
    """
    Wrapper around the FAISS library to handle vector indexing and searching.
    
    Technical Note:
    We use 'IndexFlatIP' (Inner Product). 
    Since the embeddings are L2-normalized (length = 1), the Inner Product 
    is mathematically equivalent to Cosine Similarity.
    
    Distance = 1.0 - CosineSimilarity.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.current_faiss_id_counter = 0

    def add(self, emb, fid):
        """Adds a normalized vector to the FAISS index with a specific ID."""
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]
        
        emb_copy = emb.astype('float32').copy()
        
        faiss.normalize_L2(emb_copy)
        self.index.add_with_ids(emb_copy, np.array([fid], dtype=np.int64))
        
    def search(self, query, k=1):
        """Searches for the k-nearest neighbors."""
        if query.ndim == 1:
            query = query[np.newaxis, :]
            
        query_copy = query.astype('float32').copy()
        faiss.normalize_L2(query_copy)
        
        return self.index.search(query_copy, k)

class GaitGallery:
    """
    Manages the database of known gait identities.
    Handles loading/saving, updating embeddings via EMA (Exponential Moving Average),
    and searching for matches with strict thresholding.
    """
    def __init__(self, config: GaitConfig):
        self.config = config
        self._identities: Dict[str, GaitIdentityData] = {}
        self.faiss_index = FaissIndexWrapper(config.gallery.dim)
        
        self.template_id_map: Dict[int, str] = {}
        
        if config.gallery.gallery_path.exists():
            self.load_gallery()

    def save_gallery(self):
        """Persists the gallery state to disk using Pickle."""
        try:
            state = {
                "identities": self._identities, 
                "cnt": self.faiss_index.current_faiss_id_counter
            }
            with open(self.config.gallery.gallery_path, "wb") as f:
                pickle.dump(state, f)
            logger.debug("Gallery saved successfully.")
        except Exception as e:
            logger.error(f"Error saving gallery: {e}")

    def load_gallery(self):
        """Loads the gallery state and reconstructs the FAISS index."""
        try:
            with open(self.config.gallery.gallery_path, "rb") as f:
                state = pickle.load(f)
                self._identities = state["identities"]
                self.faiss_index.current_faiss_id_counter = state["cnt"]
            
            self.faiss_index.index.reset()
            self.template_id_map.clear()
            
            tid_counter = 0
            
            for d in self._identities.values():
                templates = getattr(d, "raw_embeddings", [])
                if not templates: templates = [d.ema_embedding]
                
                for vec in templates:
                     self.faiss_index.add(vec, tid_counter)
                     
                     self.template_id_map[tid_counter] = d.identity_id
                     
                     tid_counter += 1
            
            self.faiss_index.current_faiss_id_counter = tid_counter
                     
            logger.info(f"Gallery loaded: {len(self._identities)} identities ({tid_counter} templates).")
        except Exception as e:
            logger.warning(f"No existing gallery found or file corrupted ({e}). Starting fresh.")

    def _normalize_numpy(self, x):
        """Helper to normalize a numpy array to unit length (L2 norm)."""
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            return x / norm
        return x

    def add_gait_embedding(self, identity_id, new_embedding, category="resident", confirmed=True, anthro_stats: Optional[np.ndarray] = None):
        """
        Adds or Updates an identity.
        
        If the identity exists and 'confirmed' is True, it updates the stored embedding
        using Exponential Moving Average (EMA). This allows the system to adapt
        to slight changes in a person's gait over time.
        """
        if identity_id not in self._identities:
            new_embedding = self._normalize_numpy(new_embedding)
            
            self._identities[identity_id] = GaitIdentityData(
                identity_id=identity_id, 
                faiss_id=0,
                ema_embedding=new_embedding, 
                raw_embeddings=[new_embedding], 
                num_updates=1, 
                category=category,
                anthro_mean=anthro_stats
            )
            tid = self.faiss_index.current_faiss_id_counter
            self.faiss_index.current_faiss_id_counter += 1
            
            self.faiss_index.add(new_embedding, tid)
            self.template_id_map[tid] = identity_id
            
            logger.info(f"Added new identity: {identity_id}")
            
        else:
            d = self._identities[identity_id]
            d.category = category
            
            if getattr(d, "raw_embeddings", None) is None:
                d.raw_embeddings = [d.ema_embedding]
                
            d.raw_embeddings.append(new_embedding)
            d.num_updates += 1
            
            if len(d.raw_embeddings) > 15:
                d.raw_embeddings.pop(0)
            
            alpha = self.config.gallery.ema_alpha
            d.ema_embedding = self._normalize_numpy((1 - alpha) * d.ema_embedding + alpha * new_embedding)

            if anthro_stats is not None:
                if d.anthro_mean is None:
                    d.anthro_mean = anthro_stats
                else:
                    d.anthro_mean = 0.8 * d.anthro_mean + 0.2 * anthro_stats

            
            tid = self.faiss_index.current_faiss_id_counter
            self.faiss_index.current_faiss_id_counter += 1
            
            self.faiss_index.add(new_embedding, tid)
            self.template_id_map[tid] = identity_id
            
            logger.info(f"Updated identity {identity_id} (Template #{tid})")
            
            self.save_gallery()
            return

        self.save_gallery()

    def search(self, query_embedding: np.ndarray, anthro_query: Optional[np.ndarray] = None) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
        """
        Searches the gallery with Aggregation Strategy (Max Sim per Identity).
        Returns None as match_id if no strict match, but always returns details.
        """
        total_vectors = self.faiss_index.index.ntotal
        if total_vectors == 0:
            return None, 0.0, {}

        k_search = min(50, total_vectors)
        sims, fids = self.faiss_index.search(query_embedding, k=k_search)
        
        sims_row = sims[0]
        fids_row = fids[0]

        if fids_row[0] == -1:
            return None, 0.0, {}

        debug_pids = [self.template_id_map.get(f, 'UNK') for f in fids_row if f != -1]
        logger.debug(f"🔍 DEBUG SEARCH: FIDs={fids_row[fids_row != -1]} PIDs={debug_pids}")


        candidates = {} 
        
        for i in range(len(fids_row)):
            fid = fids_row[i]
            if fid == -1: continue
            
            similarity = sims_row[i]
            distance = 1.0 - similarity
            
            pid = self.template_id_map.get(fid, "Unknown")
            
            if pid == "Unknown": continue
            
            if pid not in candidates:
                candidates[pid] = (similarity, distance)
            else:
                if similarity > candidates[pid][0]:
                    candidates[pid] = (similarity, distance)

        if anthro_query is not None:
            penalty_weight = getattr(self.config.robust, "anthro_penalty_weight", 0.5)
            
            for pid in list(candidates.keys()):
                 ident = self._identities.get(pid)
                 if ident and ident.anthro_mean is not None:
                     dist_vec = np.abs(ident.anthro_mean - anthro_query)
                     
                     geo_dist = 0.7 * dist_vec[0] + 0.3 * dist_vec[1]
                     
                     if geo_dist > 0.05:
                         original_sim, dist = candidates[pid]
                         
                         new_sim = max(0.0, original_sim - (penalty_weight * geo_dist))
                         
                         candidates[pid] = (new_sim, dist)

        ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        
        if not ranked:
            return None, 0.0, {}

        
        
        best_pid, (best_sim, best_dist) = ranked[0]
        
        details = {
            "best_pid": best_pid,
            "best_sim": best_sim,
            "best_dist": best_dist,
            "margin": 0.0
        }

        if len(ranked) > 1:
            second_pid, (second_sim, _) = ranked[1]
            margin = best_sim - second_sim
            details["second_pid"] = second_pid
            details["second_sim"] = second_sim
            details["margin"] = margin
        
        if anthro_query is not None:
            details["anthro_vec"] = anthro_query.tolist()
            if best_pid in self._identities and self._identities[best_pid].anthro_mean is not None:
                 details["match_anthro_dist"] = float(np.sum(np.abs(self._identities[best_pid].anthro_mean - anthro_query)))

        limit = self.config.thresholds.max_match_distance
        
        if best_dist > limit:
            logger.debug(f"ℹ️ Reject Weak: {best_pid} (Sim {best_sim:.4f})")
            details["status"] = "REJECT_WEAK"
            return None, best_sim, details

        if margin < self.config.thresholds.min_match_margin:
            logger.debug(f"⚠️ Low Margin ({margin:.3f}) - Passing to Engine")
            details["status"] = "LOW_MARGIN"
            return best_pid, best_sim, details
            
        return best_pid, best_sim, details

    def list_persons(self) -> List[PersonSummary]:
        return [PersonSummary(k, k, v.category, v.num_updates) for k, v in self._identities.items()]

    def delete_person(self, pid: str) -> bool:
        if pid in self._identities:
            self.faiss_index.index.remove_ids(np.array([self._identities[pid].faiss_id], dtype=np.int64))
            del self._identities[pid]
            self.save_gallery()
            return True
        return False

    def get_category(self, pid: str) -> str:
        """Returns the category (e.g., 'resident', 'visitor') of an enrolled identity."""
        if pid in self._identities:
            return self._identities[pid].category
        return "unknown"
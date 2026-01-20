
import logging
import numpy as np
import sys
from gait_subsystem.gait.config import default_gait_config
from gait_subsystem.gait.gait_gallery import GaitGallery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GalleryDiagnosis")

def diagnose():
    logger.info("loading gallery...")
    cfg = default_gait_config()
    gallery = GaitGallery(cfg)
    
    print("\nXXX=== IDENTITY STATS ===XXX")
    if not gallery._identities:
        print("GALLERY EMPTY!")
        return

    for pid, data in gallery._identities.items():
        templates = getattr(data, "raw_embeddings", [])
        print(f"ID: {pid}")
        print(f"  - Templates: {len(templates)}")
        print(f"  - Updates: {data.num_updates}")
        
        if data.anthro_mean is not None:
             print(f"  - Anthro Mean: {data.anthro_mean} (LegRatio, WidthRatio)")
        else:
             print(f"  - Anthro Mean: NONE (Critical!)")
             
        if len(templates) > 0:
            norms = [np.linalg.norm(t) for t in templates]
            print(f"  - Vector Norms (Should be ~1.0): {[f'{n:.2f}' for n in norms]}")
            
            if any(n < 0.01 for n in norms):
                print("  !!! ALERT: ZERO VECTORS DETECTED !!!")

    print("\nXXX=== FAISS INDEX STATS ===XXX")
    print(f"Total Vectors in Index: {gallery.faiss_index.index.ntotal}")
    print(f"Expected Vectors: {sum(len(d.raw_embeddings) for d in gallery._identities.values())}")
    
    if "marildo" in gallery._identities and gallery._identities["marildo"].raw_embeddings:
        marildo_vec = gallery._identities["marildo"].raw_embeddings[0]
        marildo_anthro = gallery._identities["marildo"].anthro_mean
        
        print("\nXXX=== SIMULATION: SEARCHING FOR MARILDO (Template 0) ===XXX")
        
        D, I = gallery.faiss_index.search(marildo_vec, k=50)
        print("Raw FAISS Results (Sim, FID):")
        for sim, fid in zip(D[0], I[0]):
            if fid == -1: continue
            pid = gallery.template_id_map.get(fid, "UNKNOWN")
            print(f"  > FID {fid} (PID {pid}): Sim {sim:.4f}")
            
        print("\nFull Gallery Search Logic:")
        match_id, match_sim, details = gallery.search(marildo_vec, anthro_query=marildo_anthro)
        print(f"Result: {match_id} ({match_sim:.4f})")
        print(f"Details: {details}")

if __name__ == "__main__":
    diagnose()

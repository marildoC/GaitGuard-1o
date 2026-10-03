
from __future__ import annotations

import re
import logging
from typing import Optional, List, Dict, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FaceIdentity:
    """Face gallery identity."""
    person_id: str
    name: str
    category: str
    num_templates: int


@dataclass
class GaitIdentity:
    """Gait gallery identity."""
    identity_id: str
    category: str


class MatchResult(NamedTuple):
    """Result of matching attempt."""
    face: FaceIdentity
    gait: GaitIdentity
    confidence: float
    match_type: str


class IdentityMatcher:
    """
    Intelligent matcher for face and gait identities.
    
    Strategies:
    1. EXACT: "marildo" == "marildo" (100%)
    2. PARTIAL: "marildo" in "marildo cani" (90%)
    3. FUZZY: Similar sounding names (70%+)
    
    Usage:
        matcher = IdentityMatcher()
        matches = matcher.find_matches(face_identities, gait_identities)
        
        for match in matches:
            print(f"{match.face.name} ↔ {match.gait.identity_id} ({match.confidence:.0%})")
    """
    
    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize matcher.
        
        Args:
            min_confidence: Minimum confidence to consider a match (0.0-1.0)
        """
        self.min_confidence = min_confidence
    
    def find_matches(
        self,
        face_identities: List[FaceIdentity],
        gait_identities: List[GaitIdentity]
    ) -> List[MatchResult]:
        """
        Find all matches between face and gait identities.
        
        Returns list of MatchResult, sorted by confidence (highest first).
        """
        matches = []
        
        for face in face_identities:
            for gait in gait_identities:
                confidence, match_type = self._calculate_match(
                    face.name, 
                    gait.identity_id
                )
                
                if confidence >= self.min_confidence:
                    matches.append(MatchResult(
                        face=face,
                        gait=gait,
                        confidence=confidence,
                        match_type=match_type
                    ))
        
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches
    
    def _calculate_match(
        self, 
        face_name: str, 
        gait_id: str
    ) -> Tuple[float, str]:
        """
        Calculate match confidence between face name and gait ID.
        
        Returns (confidence, match_type)
        """
        if not face_name or not gait_id:
            return 0.0, "none"
        
        face_norm = self._normalize(face_name)
        gait_norm = self._normalize(gait_id)
        
        if face_norm == gait_norm:
            return 1.0, "exact"
        
        if gait_norm in face_norm:
            ratio = len(gait_norm) / len(face_norm)
            confidence = 0.85 + (ratio * 0.10)
            return min(confidence, 0.95), "partial"
        
        if face_norm in gait_norm:
            ratio = len(face_norm) / len(gait_norm)
            confidence = 0.80 + (ratio * 0.10)
            return min(confidence, 0.90), "partial"
        
        face_words = face_norm.split()
        gait_words = gait_norm.split()
        
        for fw in face_words:
            for gw in gait_words:
                if fw == gw and len(fw) >= 3:
                    return 0.85, "word"
        
        similarity = self._string_similarity(face_norm, gait_norm)
        if similarity >= 0.75:
            return similarity * 0.9, "fuzzy"  # Cap at 0.9 for fuzzy
        
        return 0.0, "none"
    
    def _normalize(self, s: str) -> str:
        """Normalize string for comparison."""
        s = s.lower()
        s = ' '.join(s.split())
        s = re.sub(r'\s*(mr|mrs|ms|dr|prof)\.?\s*', ' ', s)
        return s.strip()
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity (0.0 to 1.0).
        
        Uses a simple approach: longest common subsequence ratio.
        """
        if not s1 or not s2:
            return 0.0
        
        lcs = self._lcs_length(s1, s2)
        
        avg_len = (len(s1) + len(s2)) / 2
        return lcs / avg_len if avg_len > 0 else 0.0
    
    def _lcs_length(self, s1: str, s2: str) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(s1), len(s2)
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


def load_face_identities() -> List[FaceIdentity]:
    """Load all identities from face gallery."""
    try:
        from identity.face_gallery import FaceGallery
        from face.config import default_face_config
        
        face_cfg = default_face_config()
        gallery = FaceGallery(face_cfg.gallery)
        
        identities = []
        for person in gallery.list_persons():
            identities.append(FaceIdentity(
                person_id=person.person_id,
                name=person.name or "",
                category=person.category,
                num_templates=person.num_templates
            ))
        
        logger.info(f"[MATCHER] Loaded {len(identities)} face identities")
        return identities
        
    except Exception as e:
        logger.error(f"[MATCHER] Failed to load face identities: {e}")
        return []


def load_gait_identities() -> List[GaitIdentity]:
    """Load all identities from gait gallery."""
    try:
        from gait_subsystem.gait.gait_gallery import GaitGallery
        from gait_subsystem.gait.config import default_gait_config
        
        gait_cfg = default_gait_config()
        gallery = GaitGallery(gait_cfg)
        
        identities = []
        for gait_id, data in gallery._identities.items():
            identities.append(GaitIdentity(
                identity_id=gait_id,
                category=data.category if hasattr(data, 'category') else "resident"
            ))
        
        logger.info(f"[MATCHER] Loaded {len(identities)} gait identities")
        return identities
        
    except Exception as e:
        logger.error(f"[MATCHER] Failed to load gait identities: {e}")
        return []


def auto_sync_registry(
    min_confidence: float = 0.7,
    dry_run: bool = False
) -> int:
    """
    Automatically sync identity registry from face and gait galleries.
    
    This is the MAIN function to call after enrollment.
    
    Args:
        min_confidence: Minimum match confidence (0.0-1.0)
        dry_run: If True, only print matches without saving
    
    Returns:
        Number of identities synced
    """
    from chimeric_identity.identity_registry import get_identity_registry
    
    print("="*70)
    print("INTELLIGENT IDENTITY SYNC")
    print("="*70)
    
    face_ids = load_face_identities()
    gait_ids = load_gait_identities()
    
    print(f"\nFace gallery: {len(face_ids)} persons")
    for f in face_ids:
        print(f"  - {f.person_id}: '{f.name}' ({f.category})")
    
    print(f"\nGait gallery: {len(gait_ids)} identities")
    for g in gait_ids:
        print(f"  - '{g.identity_id}' ({g.category})")
    
    if not face_ids or not gait_ids:
        print("\n⚠️  Need both face and gait enrollments to sync!")
        return 0
    
    matcher = IdentityMatcher(min_confidence=min_confidence)
    matches = matcher.find_matches(face_ids, gait_ids)
    
    print(f"\n{'='*70}")
    print(f"MATCHES FOUND ({len(matches)})")
    print("="*70)
    
    if not matches:
        print("\n❌ No matches found!")
        print("\nPossible reasons:")
        print("  - Names don't match between face and gait galleries")
        print("  - Try lowering min_confidence (currently {:.0%})".format(min_confidence))
        return 0
    
    synced = 0
    registry = get_identity_registry()
    
    for match in matches:
        status = "✓" if match.confidence >= 0.85 else "?"
        print(f"\n{status} MATCH ({match.confidence:.0%} - {match.match_type}):")
        print(f"   Face: {match.face.person_id} = '{match.face.name}'")
        print(f"   Gait: '{match.gait.identity_id}'")
        
        if not dry_run and match.confidence >= min_confidence:
            display_name = match.face.name if match.face.name else match.gait.identity_id
            
            registry.register_person(
                display_name=display_name,
                face_id=match.face.person_id,
                gait_id=match.gait.identity_id,
                category=match.face.category
            )
            synced += 1
            print(f"   → Registered as '{display_name}'")
    
    if dry_run:
        print(f"\n[DRY RUN] Would sync {len(matches)} identities")
    else:
        print(f"\n✓ Synced {synced} identities to registry")
    
    return synced


def manual_link(
    face_id: str,
    gait_id: str,
    display_name: Optional[str] = None,
    category: str = "resident"
) -> bool:
    """
    Manually link a face ID to a gait ID.
    
    Use this when automatic matching doesn't work.
    
    Args:
        face_id: Face gallery person ID (e.g., "p_0007")
        gait_id: Gait gallery identity ID (e.g., "marildo")
        display_name: Display name (uses face name if not provided)
        category: Person category
    
    Returns:
        True if successful
    """
    from chimeric_identity.identity_registry import get_identity_registry
    
    registry = get_identity_registry()
    
    if display_name is None:
        try:
            from identity.face_gallery import FaceGallery
            from face.config import default_face_config
            
            gallery = FaceGallery(default_face_config().gallery)
            person = gallery.get_person(face_id)
            display_name = person.name if person and person.name else gait_id
        except:
            display_name = gait_id
    
    registry.register_person(
        display_name=display_name,
        face_id=face_id,
        gait_id=gait_id,
        category=category
    )
    
    print(f"✓ Linked: '{display_name}' (face={face_id}, gait={gait_id})")
    return True



if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("""
IDENTITY MATCHER - Intelligent Face↔Gait Linking

Usage:
  python -m chimeric_identity.identity_matcher sync [--dry-run] [--min-confidence 0.7]
  python -m chimeric_identity.identity_matcher link <face_id> <gait_id> [display_name]
  python -m chimeric_identity.identity_matcher show

Examples:
  python -m chimeric_identity.identity_matcher sync
  python -m chimeric_identity.identity_matcher sync --dry-run
  python -m chimeric_identity.identity_matcher link p_0007 marildo "Marildo Cani"
""")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "sync":
        dry_run = "--dry-run" in sys.argv
        min_conf = 0.7
        
        for i, arg in enumerate(sys.argv):
            if arg == "--min-confidence" and i + 1 < len(sys.argv):
                min_conf = float(sys.argv[i + 1])
        
        auto_sync_registry(min_confidence=min_conf, dry_run=dry_run)
    
    elif cmd == "link" and len(sys.argv) >= 4:
        face_id = sys.argv[2]
        gait_id = sys.argv[3]
        display_name = sys.argv[4] if len(sys.argv) > 4 else None
        manual_link(face_id, gait_id, display_name)
    
    elif cmd == "show":
        face_ids = load_face_identities()
        gait_ids = load_gait_identities()
        
        print("\n=== FACE GALLERY ===")
        for f in face_ids:
            print(f"  {f.person_id}: '{f.name}' ({f.category})")
        
        print("\n=== GAIT GALLERY ===")
        for g in gait_ids:
            print(f"  '{g.identity_id}' ({g.category})")
        
        print("\n=== IDENTITY REGISTRY ===")
        from chimeric_identity.identity_registry import get_identity_registry
        registry = get_identity_registry()
        for p in registry.list_all():
            print(f"  {p.display_name}: face={p.face_gallery_id}, gait={p.gait_gallery_id}")
    
    else:
        print(f"Unknown command: {cmd}")

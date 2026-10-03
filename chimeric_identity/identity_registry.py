
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)



@dataclass
class PersonRecord:
    """
    Record linking a person's face and gait gallery entries.
    
    This is the key to chimeric fusion - knowing that face_gallery_id "p_0001"
    and gait_gallery_id "Francesco" refer to the SAME physical person.
    """
    display_name: str
    face_gallery_id: Optional[str] = None
    gait_gallery_id: Optional[str] = None
    category: str = "resident"  # resident / visitor / watchlist
    
    def has_face(self) -> bool:
        return self.face_gallery_id is not None
    
    def has_gait(self) -> bool:
        return self.gait_gallery_id is not None
    
    def has_both(self) -> bool:
        return self.has_face() and self.has_gait()



class IdentityRegistry:
    """
    Central registry mapping persons to their biometric gallery IDs.
    
    This solves the KEY PROBLEM in chimeric fusion:
    Face gallery uses IDs like "p_0001" while Gait gallery uses "Francesco".
    Without a registry, chimeric cannot know they're the same person!
    
    Usage:
        registry = IdentityRegistry()
        
        # During enrollment
        registry.register_person("Marildo", face_id="p_0001", gait_id="Marildo")
        
        # During recognition (chimeric fusion)
        person = registry.lookup_by_face("p_0001")  # Returns PersonRecord
        person = registry.lookup_by_gait("Marildo")  # Returns same PersonRecord
        
        # Check if face and gait refer to same person
        same = registry.are_same_person(face_id="p_0001", gait_id="Marildo")  # True
    """
    
    DEFAULT_PATH = Path("data/identity_registry.json")
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize identity registry.
        
        Args:
            registry_path: Path to JSON file for persistence
        """
        self.registry_path = registry_path or self.DEFAULT_PATH
        
        self._persons: Dict[str, PersonRecord] = {}
        
        self._face_to_person: Dict[str, str] = {}
        self._gait_to_person: Dict[str, str] = {}
        
        self._load()
        
        logger.info(
            f"[IDENTITY-REGISTRY] Initialized with {len(self._persons)} persons, "
            f"path={self.registry_path}"
        )
    
    
    def register_person(
        self,
        display_name: str,
        face_id: Optional[str] = None,
        gait_id: Optional[str] = None,
        category: str = "resident"
    ) -> PersonRecord:
        """
        Register a new person or update existing.
        
        This should be called during enrollment after adding to face/gait galleries.
        
        Args:
            display_name: Human-readable name
            face_id: ID assigned by face gallery
            gait_id: ID assigned by gait gallery
            category: Person category
        
        Returns:
            PersonRecord for the registered person
        """
        if display_name in self._persons:
            person = self._persons[display_name]
            
            if face_id:
                if person.face_gallery_id and person.face_gallery_id in self._face_to_person:
                    del self._face_to_person[person.face_gallery_id]
                person.face_gallery_id = face_id
                self._face_to_person[face_id] = display_name
            
            if gait_id:
                if person.gait_gallery_id and person.gait_gallery_id in self._gait_to_person:
                    del self._gait_to_person[person.gait_gallery_id]
                person.gait_gallery_id = gait_id
                self._gait_to_person[gait_id] = display_name
            
            person.category = category
            
            logger.info(
                f"[IDENTITY-REGISTRY] Updated: {display_name} "
                f"(face={face_id}, gait={gait_id})"
            )
        else:
            person = PersonRecord(
                display_name=display_name,
                face_gallery_id=face_id,
                gait_gallery_id=gait_id,
                category=category
            )
            self._persons[display_name] = person
            
            if face_id:
                self._face_to_person[face_id] = display_name
            if gait_id:
                self._gait_to_person[gait_id] = display_name
            
            logger.info(
                f"[IDENTITY-REGISTRY] Registered: {display_name} "
                f"(face={face_id}, gait={gait_id})"
            )
        
        self._save()
        
        return person
    
    def link_face_to_gait(self, face_id: str, gait_id: str) -> bool:
        """
        Link an existing face entry to a gait entry (by IDs).
        
        Useful when enrolling modalities separately.
        
        Returns:
            True if linked successfully, False if face_id not found
        """
        if face_id not in self._face_to_person:
            logger.warning(f"[IDENTITY-REGISTRY] Cannot link: face_id={face_id} not found")
            return False
        
        display_name = self._face_to_person[face_id]
        person = self._persons[display_name]
        
        person.gait_gallery_id = gait_id
        self._gait_to_person[gait_id] = display_name
        
        self._save()
        logger.info(f"[IDENTITY-REGISTRY] Linked gait_id={gait_id} to {display_name}")
        return True
    
    
    def lookup_by_face(self, face_id: str) -> Optional[PersonRecord]:
        """
        Look up person by face gallery ID.
        
        Args:
            face_id: ID from face recognition
        
        Returns:
            PersonRecord if found, None otherwise
        """
        if face_id is None:
            return None
        display_name = self._face_to_person.get(face_id)
        if display_name:
            return self._persons.get(display_name)
        return None
    
    def lookup_by_gait(self, gait_id: str) -> Optional[PersonRecord]:
        """
        Look up person by gait gallery ID.
        
        Args:
            gait_id: ID from gait recognition
        
        Returns:
            PersonRecord if found, None otherwise
        """
        if gait_id is None:
            return None
        display_name = self._gait_to_person.get(gait_id)
        if display_name:
            return self._persons.get(display_name)
        return None
    
    def lookup_by_name(self, display_name: str) -> Optional[PersonRecord]:
        """Look up person by display name."""
        return self._persons.get(display_name)
    
    def are_same_person(
        self, 
        face_id: Optional[str], 
        gait_id: Optional[str]
    ) -> bool:
        """
        Check if face_id and gait_id refer to the same person.
        
        THIS IS THE KEY METHOD FOR CHIMERIC FUSION!
        
        Args:
            face_id: ID from face recognition
            gait_id: ID from gait recognition
        
        Returns:
            True if same person, False otherwise
        """
        if face_id is None or gait_id is None:
            return False
        
        face_person = self._face_to_person.get(face_id)
        gait_person = self._gait_to_person.get(gait_id)
        
        if face_person is None or gait_person is None:
            return face_id == gait_id
        
        return face_person == gait_person
    
    def get_display_name(
        self,
        face_id: Optional[str] = None,
        gait_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get display name from either face or gait ID.
        
        Useful for showing human-readable name in UI.
        """
        if face_id:
            person = self.lookup_by_face(face_id)
            if person:
                return person.display_name
        
        if gait_id:
            person = self.lookup_by_gait(gait_id)
            if person:
                return person.display_name
        
        return face_id or gait_id
    
    
    def _save(self) -> None:
        """Save registry to JSON file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": 1,
                "persons": {
                    name: asdict(person)
                    for name, person in self._persons.items()
                }
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[IDENTITY-REGISTRY] Saved {len(self._persons)} persons")
            
        except Exception as e:
            logger.error(f"[IDENTITY-REGISTRY] Save failed: {e}")
    
    def _load(self) -> None:
        """Load registry from JSON file."""
        if not self.registry_path.exists():
            logger.info(f"[IDENTITY-REGISTRY] No existing registry at {self.registry_path}")
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            
            persons_data = data.get("persons", {})
            for name, person_dict in persons_data.items():
                person = PersonRecord(
                    display_name=person_dict.get("display_name", name),
                    face_gallery_id=person_dict.get("face_gallery_id"),
                    gait_gallery_id=person_dict.get("gait_gallery_id"),
                    category=person_dict.get("category", "resident")
                )
                self._persons[name] = person
                
                if person.face_gallery_id:
                    self._face_to_person[person.face_gallery_id] = name
                if person.gait_gallery_id:
                    self._gait_to_person[person.gait_gallery_id] = name
            
            logger.info(
                f"[IDENTITY-REGISTRY] Loaded {len(self._persons)} persons "
                f"from {self.registry_path}"
            )
            
        except Exception as e:
            logger.error(f"[IDENTITY-REGISTRY] Load failed: {e}")
    
    
    def list_all(self) -> List[PersonRecord]:
        """List all registered persons."""
        return list(self._persons.values())
    
    def count(self) -> int:
        """Count registered persons."""
        return len(self._persons)
    
    def remove_person(self, display_name: str) -> bool:
        """Remove a person from registry."""
        if display_name not in self._persons:
            return False
        
        person = self._persons[display_name]
        
        if person.face_gallery_id:
            self._face_to_person.pop(person.face_gallery_id, None)
        if person.gait_gallery_id:
            self._gait_to_person.pop(person.gait_gallery_id, None)
        
        del self._persons[display_name]
        self._save()
        
        logger.info(f"[IDENTITY-REGISTRY] Removed: {display_name}")
        return True
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._persons.clear()
        self._face_to_person.clear()
        self._gait_to_person.clear()
        self._save()
        logger.info("[IDENTITY-REGISTRY] Cleared all registrations")



_global_registry: Optional[IdentityRegistry] = None


def get_identity_registry(registry_path: Optional[Path] = None) -> IdentityRegistry:
    """
    Get global identity registry singleton.
    
    Use this in chimeric fusion to access the registry.
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = IdentityRegistry(registry_path)
    
    return _global_registry



if __name__ == "__main__":
    import sys
    
    registry = get_identity_registry()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python identity_registry.py list")
        print("  python identity_registry.py add <name> [--face <id>] [--gait <id>]")
        print("  python identity_registry.py remove <name>")
        print("  python identity_registry.py check <face_id> <gait_id>")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "list":
        persons = registry.list_all()
        if not persons:
            print("No persons registered")
        else:
            print(f"Registered persons ({len(persons)}):")
            for p in persons:
                print(f"  {p.display_name}: face={p.face_gallery_id}, gait={p.gait_gallery_id}")
    
    elif cmd == "add" and len(sys.argv) >= 3:
        name = sys.argv[2]
        face_id = None
        gait_id = None
        
        for i, arg in enumerate(sys.argv):
            if arg == "--face" and i + 1 < len(sys.argv):
                face_id = sys.argv[i + 1]
            if arg == "--gait" and i + 1 < len(sys.argv):
                gait_id = sys.argv[i + 1]
        
        registry.register_person(name, face_id=face_id, gait_id=gait_id)
        print(f"Registered: {name}")
    
    elif cmd == "remove" and len(sys.argv) >= 3:
        name = sys.argv[2]
        if registry.remove_person(name):
            print(f"Removed: {name}")
        else:
            print(f"Not found: {name}")
    
    elif cmd == "check" and len(sys.argv) >= 4:
        face_id = sys.argv[2]
        gait_id = sys.argv[3]
        same = registry.are_same_person(face_id, gait_id)
        print(f"face_id={face_id}, gait_id={gait_id} -> Same person: {same}")
    
    else:
        print(f"Unknown command: {cmd}")

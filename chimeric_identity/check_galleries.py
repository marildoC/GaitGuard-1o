
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_face_gallery():
    """Check face gallery status."""
    print("\n" + "="*60)
    print("FACE GALLERY STATUS")
    print("="*60)
    
    try:
        from identity.face_gallery import FaceGallery
        from face.config import default_face_config
        
        face_cfg = default_face_config()
        gallery_path = face_cfg.gallery.gallery_path
        
        print(f"Gallery path: {gallery_path}")
        print(f"File exists: {Path(gallery_path).exists()}")
        
        if Path(gallery_path).exists():
            file_size = Path(gallery_path).stat().st_size
            print(f"File size: {file_size:,} bytes")
        
        gallery = FaceGallery(face_cfg.gallery)
        
        persons = gallery.list_persons()
        print(f"\nEnrolled persons: {len(persons)}")
        
        if persons:
            print("\nPersons in gallery:")
            for p in persons:
                print(f"  - {p.person_id}: name='{p.name}', category={p.category}, templates={p.num_templates}")
        else:
            print("  (No persons enrolled)")
            
        return persons
        
    except Exception as e:
        print(f"ERROR: {e}")
        return []


def check_gait_gallery():
    """Check gait gallery status."""
    print("\n" + "="*60)
    print("GAIT GALLERY STATUS")
    print("="*60)
    
    try:
        from gait_subsystem.gait.gait_gallery import GaitGallery
        from gait_subsystem.gait.config import default_gait_config
        
        gait_cfg = default_gait_config()
        gallery_path = gait_cfg.gallery.gallery_path
        
        print(f"Gallery path: {gallery_path}")
        print(f"File exists: {gallery_path.exists()}")
        
        if gallery_path.exists():
            file_size = gallery_path.stat().st_size
            print(f"File size: {file_size:,} bytes")
        
        gallery = GaitGallery(gait_cfg)
        
        identities = list(gallery._identities.keys())
        print(f"\nEnrolled identities: {len(identities)}")
        
        if identities:
            print("\nIdentities in gallery:")
            for gait_id in identities:
                data = gallery._identities[gait_id]
                print(f"  - {gait_id}: category={data.category}, updates={data.num_updates}")
        else:
            print("  (No identities enrolled)")
            
        return identities
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


def check_identity_registry():
    """Check identity registry status."""
    print("\n" + "="*60)
    print("IDENTITY REGISTRY STATUS")
    print("="*60)
    
    try:
        from chimeric_identity.identity_registry import get_identity_registry
        
        registry = get_identity_registry()
        
        print(f"Registry path: {registry.registry_path}")
        print(f"File exists: {registry.registry_path.exists()}")
        
        persons = registry.list_all()
        print(f"\nRegistered persons: {len(persons)}")
        
        if persons:
            print("\nPersons in registry:")
            for p in persons:
                face_status = "✓" if p.has_face() else "✗"
                gait_status = "✓" if p.has_gait() else "✗"
                print(f"  - {p.display_name}:")
                print(f"      Face [{face_status}]: {p.face_gallery_id}")
                print(f"      Gait [{gait_status}]: {p.gait_gallery_id}")
        else:
            print("  (No persons registered)")
        
        return persons
        
    except Exception as e:
        print(f"ERROR: {e}")
        return []


def find_matching_names(face_persons, gait_identities):
    """Find persons that exist in both galleries with matching names."""
    print("\n" + "="*60)
    print("MATCHING NAMES ANALYSIS")
    print("="*60)
    
    face_names = {}
    for p in face_persons:
        if p.name:
            face_names[p.name.lower()] = p.person_id
    
    matches = []
    for gait_id in gait_identities:
        gait_name_lower = gait_id.lower()
        if gait_name_lower in face_names:
            matches.append({
                'name': gait_id,
                'face_id': face_names[gait_name_lower],
                'gait_id': gait_id
            })
    
    if matches:
        print(f"\nFound {len(matches)} matching names:")
        for m in matches:
            print(f"  - '{m['name']}': face={m['face_id']}, gait={m['gait_id']}")
        print("\nThese can be auto-synced to identity registry!")
    else:
        print("\nNo matching names found between face and gait galleries.")
        print("You need to enroll persons in BOTH galleries with same name.")
    
    return matches


def main():
    print("="*60)
    print("CHIMERIC GALLERY STATUS CHECK")
    print("="*60)
    
    face_persons = check_face_gallery()
    gait_identities = check_gait_gallery()
    registry_persons = check_identity_registry()
    
    matches = find_matching_names(face_persons, gait_identities)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if not face_persons:
        print("\n⚠️  Face gallery is empty or cannot be read.")
        print("   → Check if GAITGUARD_FACE_KEY environment variable is set")
        print("   → Or enroll faces using: python -m identity.enrollment_cli")
    
    if not gait_identities:
        print("\n⚠️  Gait gallery is empty.")
        print("   → Enroll gaits using: python -m gait_subsystem.enrollment_cli")
    
    if not registry_persons and matches:
        print("\n💡 Matches found but registry is empty!")
        print("   → Run sync: python -m chimeric_identity.enrollment_integration sync")
    
    if face_persons and gait_identities and not matches:
        print("\n⚠️  Face and gait galleries have different names!")
        print("   → Manually register in registry with matching names")
        print("   → Or re-enroll with consistent naming")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

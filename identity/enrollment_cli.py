
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from face.config import default_face_config
from face.detector_align import FaceDetectorAligner
from face.quality import compute_full_quality
from face.embedder import FaceEmbedder
from identity.face_gallery import FaceGallery, PersonSummary
from identity.crypto import CryptoConfig
from identity.enrollment_multiview_guide import (
    EnrollmentMultiViewGuide,
    GuidedCaptureConfig,
)

from identity.watchlist_enroll import (
    create_watchlist_person_draft,
    process_image,
    commit_watchlist_enrollment,
    FaceTemplateProposal,
    MultiFaceImageResult,
    NoFaceResult,
    ImageReadError,
)

logger = logging.getLogger(__name__)




def _open_camera(
    index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> cv2.VideoCapture:
    """
    Open a camera device with basic configuration.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _setup_logging(level: int = logging.INFO) -> None:
    """
    Ensure root logger is configured for CLI use.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)




def _handle_locked_gallery(gallery: FaceGallery) -> bool:
    """
    If the FaceGallery instance is locked due to a previous load error
    (likely key mismatch or corruption), print a clear explanation and
    return True. Caller should then abort the command.

    Returns False if gallery is OK to use.
    """
    if not gallery.locked_due_to_error:
        return False

    cfg = default_face_config()
    path = cfg.gallery.gallery_path
    env_var = CryptoConfig.env_var
    err = gallery.last_load_error or "unknown error"

    msg = f"""
Face gallery could not be decrypted and is now locked for this process.

  Path      : {path}
  Env var   : {env_var}
  Last error: {err}

Most likely causes:
  - The encryption key in the environment ({env_var}) is different from
    the one used when this gallery was created, OR
  - The gallery file is corrupted.

The on-disk file was NOT modified.

To reset and start with a fresh (empty) gallery, run:

  python -m identity.enrollment_cli reset-gallery

WARNING: this will DELETE the current gallery file permanently.
"""
    print(msg.strip() + "\n")
    return True


def _generate_person_id(gallery: FaceGallery) -> str:
    """
    Generate a new person_id like 'p_0001', 'p_0002', ...
    based purely on existing keys in gallery.persons.

    This avoids any dependency on internal gallery allocation logic
    and stays compatible with existing IDs.
    """
    max_idx = 0
    for pid in gallery.persons.keys():
        if not isinstance(pid, str):
            continue
        if not pid.startswith("p_"):
            continue
        tail = pid[2:]
        try:
            n = int(tail)
        except ValueError:
            continue
        if n > max_idx:
            max_idx = n

    return f"p_{max_idx + 1:04d}"


# ---------------------------------------------------------------------------
# Enrollment logic (classic 2D)
# ---------------------------------------------------------------------------


def _ensure_embedding(emb: np.ndarray) -> np.ndarray:
    """
    Ensure a 1-D float32, L2-normalised embedding.

    buffalo_l usually outputs L2-normalised 512-D vectors, but we
    enforce normalisation again for safety and consistency.
    """
    e = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(e))
    if norm > 1e-6:
        e /= norm
    else:
        e[:] = 0.0
    return e


def _collect_face_embeddings_interactive(
    detector: FaceDetectorAligner,
    max_samples: int,
    camera_index: int,
) -> List[np.ndarray]:
    """
    Interactive capture loop (classic 2D enrollment):

    - Opens webcam.
    - On SPACE: runs face detector (buffalo_l) on the full frame,
      picks the best-quality face, and stores its embedding.
    - On ENTER: finish and return collected embeddings.
    - On ESC: abort and return empty list.
    """
    cfg = default_face_config()
    th = cfg.thresholds

    # Prefer new config field if present, otherwise fall back to old one.
    min_q_enroll = getattr(
        th,
        "min_quality_enroll",
        getattr(th, "min_quality_for_embed", 0.5),
    )

    cap = _open_camera(camera_index)
    logger.info(
        "Camera opened on index %d. Press SPACE to capture, ENTER to finish, ESC to abort.",
        camera_index,
    )

    embeddings: List[np.ndarray] = []
    last_info = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Failed to read frame from camera")
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # HUD text
            if last_info:
                cv2.putText(
                    display,
                    last_info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.putText(
                display,
                f"samples: {len(embeddings)}/{max_samples}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("GaitGuard Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                logger.info("Enrollment aborted by user (ESC).")
                embeddings.clear()
                break

            if key == 13:  # ENTER
                logger.info("Enrollment finished by user (ENTER).")
                break

            if key == 32:  # SPACE -> capture
                logger.info("Capture requested (SPACE). Running face detector.")
                last_info = "Detecting face..."

                # Run buffalo_l-based detector on the full frame.
                candidates = detector.detect_and_align(frame)
                if not candidates:
                    last_info = "No face detected. Try again."
                    logger.info(last_info)
                    continue

                # Select best candidate by our quality metric.
                best_cand = None
                best_q = 0.0
                for cand in candidates:
                    q = compute_full_quality(
                        image=frame,
                        bbox=cand.bbox,
                        det_score=cand.det_score,
                        yaw=cand.yaw,
                        pitch=cand.pitch,
                        cfg=cfg,
                    )
                    if q > best_q:
                        best_q = q
                        best_cand = cand

                if best_cand is None:
                    last_info = "Face detection failed. Try again."
                    logger.info(last_info)
                    continue

                if best_q < min_q_enroll:
                    last_info = (
                        f"Face quality too low (q={best_q:.2f}). "
                        "Move closer / look at camera."
                    )
                    logger.info(last_info)
                    continue

                if best_cand.embedding is None:
                    last_info = "Detected face has no embedding. See logs."
                    logger.error("Best candidate returned without embedding.")
                    continue

                try:
                    emb = _ensure_embedding(best_cand.embedding)
                except Exception as exc:
                    logger.exception("Embedding normalisation failed: %s", exc)
                    last_info = "Embedding failed. See logs."
                    continue

                embeddings.append(emb)
                last_info = f"Captured sample #{len(embeddings)} (q={best_q:.2f})"
                logger.info(last_info)

                if len(embeddings) >= max_samples:
                    logger.info("Reached max_samples=%d", max_samples)
                    break

        return embeddings
    finally:
        cap.release()
        cv2.destroyAllWindows()


def cmd_enroll(args: argparse.Namespace) -> None:
    """
    Classic 2D enroll: enroll a new person OR append templates
    to an existing one based on multiple captures from webcam.
    """
    _setup_logging()
    cfg = default_face_config()
    gallery = FaceGallery(cfg.gallery)

    # Respect Wave 2.1 hardening: abort if gallery is locked.
    if _handle_locked_gallery(gallery):
        return

    detector = FaceDetectorAligner(cfg)

    logger.info("Starting enrollment process (classic 2D).")

    existing_id = (args.existing or "").strip()

    if existing_id:
        # Append new templates to an existing person.
        person = gallery.persons.get(existing_id)
        if person is None:
            print(f"Existing person_id '{existing_id}' not found in gallery.")
            return

        print(f"Appending templates to existing person '{existing_id}'.")
        print(f"  current name     : {person.name or '(none)'}")
        print(f"  current category : {person.category}")
        print(f"  current templates: {len(person.templates)}")

        # Optional metadata updates; default to empty (no change).
        name = args.name or ""
        category = args.category or ""

        # Condition label for these new templates.
        condition = args.condition or input(
            "Condition for new templates [neutral/glasses/cap/etc] (default=neutral): "
        ).strip()
        if not condition:
            condition = "neutral"

        notes = args.notes or input("Notes for these templates (optional): ").strip()

        max_samples = args.samples
        camera_index = args.camera

        embs = _collect_face_embeddings_interactive(
            detector=detector,
            max_samples=max_samples,
            camera_index=camera_index,
        )

        if not embs:
            logger.warning("No embeddings collected. Gallery not modified.")
            return

        before = len(person.templates)
        for emb in embs:
            meta = {"source": "cli_enroll"}
            if notes:
                meta["notes"] = notes
            gallery.add_template(
                existing_id,
                embedding=emb,
                condition=condition,
                metadata=meta,
            )

        # Optional metadata updates (rename / recategorise) in one go.
        meta_updates = {}
        if category:
            meta_updates["category"] = category
        if name:
            meta_updates["name"] = name
        if meta_updates:
            gallery.update_metadata(existing_id, **meta_updates)

        gallery.save()

        after = len(gallery.persons[existing_id].templates)
        print("\nEnrollment (append) complete.")
        print(f"  person_id   : {existing_id}")
        print(f"  name        : {gallery.persons[existing_id].name or '(none)'}")
        print(f"  category    : {gallery.persons[existing_id].category}")
        print(f"  condition   : {condition}")
        print(f"  added       : {after - before} templates")
        print(f"  total now   : {after}")
        return

    # ------------------------------------------------------------------
    # New person enrollment (no --existing)
    # ------------------------------------------------------------------
    name = args.name or input("Name (optional, can be empty): ").strip()
    category = args.category or input(
        "Category [resident/visitor/watchlist] (default=resident): "
    ).strip()
    if not category:
        category = "resident"

    notes = args.notes or input("Notes (optional): ").strip()
    condition = args.condition or input(
        "Condition [neutral/glasses/cap/etc] (default=neutral): "
    ).strip()
    if not condition:
        condition = "neutral"

    max_samples = args.samples
    camera_index = args.camera

    embs = _collect_face_embeddings_interactive(
        detector=detector,
        max_samples=max_samples,
        camera_index=camera_index,
    )

    if not embs:
        logger.warning("No embeddings collected. Person will not be enrolled.")
        return

    emb_array = np.stack(embs, axis=0)
    metadata = {}
    if notes:
        metadata["notes"] = notes

    person_id = gallery.enroll_person(
        embeddings=emb_array,
        name=name if name else None,
        category=category,
        condition=condition,
        notes=notes if notes else None,
        metadata=metadata,
    )
    gallery.save()

    print("\nEnrollment complete.")
    print(f"  person_id : {person_id}")
    print(f"  name      : {name or '(none)'}")
    print(f"  category  : {category}")
    print(f"  condition : {condition}")
    print(f"  samples   : {emb_array.shape[0]}")


# ---------------------------------------------------------------------------
# Guided 3D-aware enrollment (multi-view)
# ---------------------------------------------------------------------------


def cmd_guided_enroll(args: argparse.Namespace) -> None:
    """
    Guided 5-pose enrollment:

    FRONT / LEFT / RIGHT / UP / DOWN

    Uses the unified Phase-2A face pipeline (detector + quality) and
    stores pose-aware templates with rich metadata for 3D/multi-view.
    """
    _setup_logging()
    cfg = default_face_config()
    gallery = FaceGallery(cfg.gallery)

    if _handle_locked_gallery(gallery):
        return

    detector = FaceDetectorAligner(cfg)
    embedder = FaceEmbedder(cfg)

    # Person identity info
    name = args.name or input("Name (optional): ").strip()
    surname = args.surname or input("Surname (optional): ").strip()

    # Category (resident / visitor / watchlist)
    category = args.category or input(
        "Category [resident/visitor/watchlist] (default=resident): "
    ).strip()
    if not category:
        category = "resident"

    # Condition & notes (for templates metadata)
    condition = args.condition or input(
        "Condition [neutral/glasses/cap/etc] (default=neutral): "
    ).strip()
    if not condition:
        condition = "neutral"

    notes = args.notes or input("Notes (optional): ").strip()

    # Camera & capture behaviour
    camera_index = args.camera
    frames_per_pose = args.frames_per_pose
    min_q = args.min_quality
    display = not args.no_display

    # Decide person_id:
    #   - If provided explicitly, use it.
    #   - Otherwise generate a new p_XXXX following existing pattern.
    person_id = (args.person_id or "").strip()
    if not person_id:
        person_id = _generate_person_id(gallery)
        logger.info("Generated new person_id for guided enrollment: %s", person_id)

    logger.info(
        "Starting guided multi-view enrollment for %s (name=%s %s, category=%s, condition=%s)",
        person_id,
        name,
        surname,
        category,
        condition,
    )

    capture_cfg = GuidedCaptureConfig(
        num_frames=frames_per_pose,
        min_quality=min_q,
        yaw_tolerance_deg=args.yaw_tolerance,
        pitch_tolerance_deg=args.pitch_tolerance,
        display=display,
    )

    guide = EnrollmentMultiViewGuide(
        gallery=gallery,
        aligner=detector,
        embedder=embedder,
        capture_cfg=capture_cfg,
    )

    # extra metadata per template (merged in enrollment_multiview_guide)
    extra_meta = {
        "source": "cli_guided_enroll",
        "condition": condition,
    }
    if notes:
        extra_meta["notes"] = notes

    # Run the guided flow (this writes templates into gallery.persons[person_id])
    guide.guided_enroll(
        person_id=person_id,
        name=name,
        surname=surname,
        camera_index=camera_index,
        extra_metadata=extra_meta,
    )

    # Ensure category and name are set on the person entry.
    meta_updates = {
        "category": category,
    }
    if name or surname:
        # Let the gallery decide how to display; we pass a single 'name' field.
        full_name = (name + " " + surname).strip() or None
        if full_name:
            meta_updates["name"] = full_name

    gallery.update_metadata(person_id, **meta_updates)
    gallery.save()

    print("\nGuided multi-view enrollment complete.")
    print(f"  person_id : {person_id}")
    print(f"  name      : {(name + ' ' + surname).strip() or '(none)'}")
    print(f"  category  : {category}")
    print(f"  condition : {condition}")
    print("  Mode      : 5-pose guided (3D-aware)")


# ---------------------------------------------------------------------------
# Image-based WATCHLIST enrollment (new)
# ---------------------------------------------------------------------------


def _print_watchlist_template_summary(
    proposals: List[FaceTemplateProposal],
) -> None:
    """
    Helper for CLI: print a compact table of accepted templates
    for a watchlist person.
    """
    if not proposals:
        print("No templates accepted yet.")
        return

    print()
    print(f"{'idx':>3} {'file':<24} {'pose':<7} {'q':>5} {'level':<6} {'area':>8} {'warnings':<0}")
    print("-" * 72)
    for idx, p in enumerate(proposals):
        fname = p.image_path.name
        if len(fname) > 24:
            fname = fname[:21] + "..."
        warns = "; ".join(p.warnings) if p.warnings else ""
        print(
            f"{idx:>3} {fname:<24} {p.pose_bin:<7} {p.quality:>5.2f} "
            f"{p.quality_level:<6} {p.face_area:>8} {warns}"
        )
    print()


def cmd_enroll_watchlist(args: argparse.Namespace) -> None:
    """
    Enroll a WATCHLIST person from one or more external images.

    This uses the same face backend (detector + embedder + quality) as
    live enrollment, but the source is images on disk.

    Invariants:
      - We NEVER write partial data.
      - The gallery is modified only after the user explicitly confirms.
      - Even LOW-quality faces are allowed, with clear warnings.
    """
    _setup_logging()
    cfg = default_face_config()
    gallery = FaceGallery(cfg.gallery)

    if _handle_locked_gallery(gallery):
        return

    # Reuse backends so we don't reload models for every image.
    detector = FaceDetectorAligner(cfg)
    embedder = FaceEmbedder(cfg)

    print("=== Image-based WATCHLIST enrollment ===\n")

    # Person-level metadata
    name = args.name or input("Name (required): ").strip()
    while not name:
        name = input("Name cannot be empty. Please enter name: ").strip()

    surname = args.surname or input("Surname (optional): ").strip()
    country = args.country or input("Country (optional, ISO code or text): ").strip()
    notes = args.notes or input(
        "Description / notes (age, gender, case, etc.) (optional): "
    ).strip()

    # Create an in-memory draft; this does NOT touch the gallery yet.
    draft = create_watchlist_person_draft(
        gallery,
        name=name,
        surname=surname or None,
        country=country or None,
        notes=notes or None,
    )

    print("\nCreated watchlist draft:")
    print(f"  person_id : {draft.person_id}")
    print(f"  name      : {draft.full_name()}")
    print(f"  country   : {draft.country or '(none)'}")
    print(f"  notes     : {draft.notes or '(none)'}")
    print("Now add one or more images of this person.\n")

    accepted: List[FaceTemplateProposal] = []

    # -------------------------------
    # Image loop: add proposals
    # -------------------------------
    while True:
        if not accepted:
            prompt = (
                "Enter path to image (at least one required, "
                "or ENTER to cancel): "
            )
        else:
            prompt = "Enter path to image (or just ENTER to finish adding): "

        path_str = input(prompt).strip()
        if not path_str:
            if not accepted:
                # No templates yet → give user a chance to cancel cleanly.
                cancel = input(
                    "No images accepted yet. Cancel enrollment? [y/N]: "
                ).strip().lower()
                if cancel in ("y", "yes"):
                    print(
                        "Watchlist enrollment cancelled. Gallery was not modified."
                    )
                    return
                else:
                    # continue loop to force at least one accepted template
                    continue
            else:
                # We already have some templates; break to summary/confirm.
                break

        img_path = Path(path_str)
        if not img_path.exists():
            print(f"File not found: {img_path}")
            continue

        # First pass: detect faces without specifying index.
        outcome = process_image(
            draft,
            img_path,
            cfg=cfg,
            detector=detector,
            embedder=embedder,
        )

        # Handle error / no face
        if isinstance(outcome, ImageReadError):
            print(f"[ERROR] Could not read image: {img_path}")
            continue

        if isinstance(outcome, NoFaceResult):
            print(f"[INFO] No usable face found in image: {img_path}")
            if outcome.reason == "face_too_small":
                print("       (face too small; try a closer / higher-resolution image)")
            continue

        # Handle multiple faces → display options, ask which index to use
        if isinstance(outcome, MultiFaceImageResult):
            print(f"Image {img_path} contains {len(outcome.faces)} faces:")
            print(f"{'idx':>3} {'bbox(x1,y1,x2,y2)':<26} {'area':>8} {'yaw':>6} {'pitch':>6} {'score':>7}")
            print("-" * 72)
            for f in outcome.faces:
                x1, y1, x2, y2 = f.bbox_xyxy
                print(
                    f"{f.index:>3} "
                    f"({x1:4d},{y1:4d},{x2:4d},{y2:4d}) "
                    f"{f.face_area:>8} "
                    f"{f.yaw_deg:>6.1f} "
                    f"{f.pitch_deg:>6.1f} "
                    f"{f.det_score:>7.2f}"
                )

            while True:
                idx_str = input(
                    "Select face index to use (or ENTER to skip this image): "
                ).strip()
                if not idx_str:
                    print("Skipping this image.")
                    outcome = None
                    break
                try:
                    idx = int(idx_str)
                except ValueError:
                    print("Invalid index; please enter an integer.")
                    continue

                proposal_or_other = process_image(
                    draft,
                    img_path,
                    face_index=idx,
                    cfg=cfg,
                    detector=detector,
                    embedder=embedder,
                )
                if isinstance(proposal_or_other, FaceTemplateProposal):
                    outcome = proposal_or_other
                    break
                elif isinstance(proposal_or_other, NoFaceResult):
                    print(
                        f"[INFO] Selected face index {idx} is invalid "
                        f"or too small; try another index."
                    )
                    outcome = None
                    # loop again asking for index
                else:
                    print(
                        "[WARN] Unexpected outcome when re-processing selected face. "
                        "Skipping this image."
                    )
                    outcome = None
                    break

            if outcome is None:
                continue

        # At this point, outcome must be a FaceTemplateProposal
        if not isinstance(outcome, FaceTemplateProposal):
            print(
                "[WARN] Unexpected processing result. "
                "Skipping this image; see logs for details."
            )
            continue

        prop: FaceTemplateProposal = outcome

        # Show summary for this template
        print("\nProposed template from image:")
        print(f"  file        : {prop.image_path}")
        print(f"  pose_bin    : {prop.pose_bin}")
        print(f"  yaw/pitch   : {prop.yaw_deg:.1f} / {prop.pitch_deg:.1f}")
        print(f"  quality     : {prop.quality:.2f} ({prop.quality_level})")
        print(f"  face area   : {prop.face_area}")
        print(f"  det_score   : {prop.det_score:.2f}")
        if prop.warnings:
            print("  warnings    :")
            for w in prop.warnings:
                print(f"    - {w}")

        # Let user decide whether to accept low-quality images
        accept_str = input("Accept this image as a template? [Y/n]: ").strip().lower()
        if accept_str in ("", "y", "yes"):
            accepted.append(prop)
            print(f"Template accepted (total now: {len(accepted)}).")
        else:
            print("Template discarded.")

    # -------------------------------
    # Final summary / edit / commit
    # -------------------------------
    while True:
        print("\nCurrent watchlist draft:")
        print(f"  person_id : {draft.person_id}")
        print(f"  name      : {draft.full_name()}")
        print(f"  country   : {draft.country or '(none)'}")
        print(f"  notes     : {draft.notes or '(none)'}")
        _print_watchlist_template_summary(accepted)

        print("Options:")
        print("  [C] Confirm and save")
        print("  [R] Remove a template by index")
        print("  [E] Edit person metadata (name/surname/country/notes)")
        print("  [X] Cancel (discard everything)")
        choice = input("Choose an option [C/R/E/X]: ").strip().lower()

        if choice in ("c", ""):
            # Confirm & save
            break
        elif choice == "x":
            print(
                "Watchlist enrollment cancelled. "
                "No changes were written to the gallery."
            )
            return
        elif choice == "r":
            if not accepted:
                print("No templates to remove.")
                continue
            idx_str = input(
                "Enter template index to remove (or ENTER to cancel): "
            ).strip()
            if not idx_str:
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                print("Invalid index.")
                continue
            if idx < 0 or idx >= len(accepted):
                print("Index out of range.")
                continue
            removed = accepted.pop(idx)
            print(f"Removed template for file: {removed.image_path.name}")
            if not accepted:
                print(
                    "Warning: no templates left. You must add at least one "
                    "image again or cancel the enrollment."
                )
                # Fall back to image loop again
                # but to keep flow simple, we just let them cancel or restart.
            continue
        elif choice == "e":
            # Edit person-level metadata
            new_name = input(
                f"Name [{draft.name}]: "
            ).strip()
            if new_name:
                draft.name = new_name

            new_surname = input(
                f"Surname [{draft.surname or ''}]: "
            ).strip()
            if new_surname or draft.surname:
                draft.surname = new_surname or draft.surname

            new_country = input(
                f"Country [{draft.country or ''}]: "
            ).strip()
            if new_country or draft.country:
                draft.country = new_country or draft.country

            new_notes = input(
                f"Notes [{draft.notes or ''}]: "
            ).strip()
            if new_notes or draft.notes:
                draft.notes = new_notes or draft.notes
            continue
        else:
            print("Invalid option. Please choose C, R, E or X.")
            continue

    # Confirmed: commit to gallery
    if not accepted:
        print(
            "No templates available to save. "
            "Watchlist person will NOT be enrolled."
        )
        return

    copy_images = not args.no_copy_images
    raw_root = args.raw_root or None

    person_id = commit_watchlist_enrollment(
        gallery,
        draft,
        accepted,
        copy_images=copy_images,
        raw_root=raw_root,
    )

    print("\nWatchlist enrollment complete.")
    print(f"  person_id : {person_id}")
    print(f"  name      : {draft.full_name()}")
    print(f"  category  : watchlist")
    print(f"  templates : {len(accepted)}")
    print(
        "Person is now stored in the encrypted gallery. "
        "Runtime identity engine will detect them like any other watchlist identity."
    )


# ---------------------------------------------------------------------------
# List / delete commands
# ---------------------------------------------------------------------------


def _print_persons(persons: List[PersonSummary]) -> None:
    if not persons:
        print("No persons enrolled in the gallery.")
        return

    print(f"{'person_id':<16} {'category':<10} {'name':<24} {'templates':>9}")
    print("-" * 64)
    for p in persons:
        name = p.name or ""
        print(f"{p.person_id:<16} {p.category:<10} {name:<24} {p.num_templates:>9}")


def cmd_list(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = default_face_config()
    gallery = FaceGallery(cfg.gallery)

    if _handle_locked_gallery(gallery):
        return

    persons = gallery.list_persons()
    _print_persons(persons)


def cmd_delete(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = default_face_config()
    gallery = FaceGallery(cfg.gallery)

    if _handle_locked_gallery(gallery):
        return

    person_id = args.person_id or input("person_id to delete: ").strip()
    if not person_id:
        print("No person_id provided.")
        return

    confirm = input(
        f"Are you sure you want to delete '{person_id}'? [y/N]: "
    ).strip().lower()
    if confirm not in ("y", "yes"):
        print("Deletion cancelled.")
        return

    ok = gallery.delete_person(person_id)
    if not ok:
        print(f"Person '{person_id}' not found.")
        return

    gallery.save()
    print(f"Person '{person_id}' deleted.")


# ---------------------------------------------------------------------------
# Rename / category update commands (Wave 2.2 CRUD)
# ---------------------------------------------------------------------------


def cmd_rename(args: argparse.Namespace) -> None:
    """
    Rename an existing person in the gallery.
    """
    _setup_logging()
    cfg = default_face_config()
    gallery = FaceGallery(cfg.gallery)

    if _handle_locked_gallery(gallery):
        return

    person_id = args.person_id
    person = gallery.persons.get(person_id)
    if person is None:
        print(f"Person '{person_id}' not found.")
        return

    new_name = args.name or input(
        f"New name for '{person_id}' (current: {person.name or '(none)'}): "
    ).strip()
    if not new_name:
        print("Empty name; nothing changed.")
        return

    gallery.update_metadata(person_id, name=new_name)
    gallery.save()
    print(f"Person '{person_id}' renamed to '{new_name}'.")


def cmd_set_category(args: argparse.Namespace) -> None:
    """
    Update category (resident / visitor / watchlist) for an existing person.
    """
    _setup_logging()
    cfg = default_face_config()
    gallery = FaceGallery(cfg.gallery)

    if _handle_locked_gallery(gallery):
        return

    person_id = args.person_id
    person = gallery.persons.get(person_id)
    if person is None:
        print(f"Person '{person_id}' not found.")
        return

    new_cat = (args.category or "").strip().lower()
    if not new_cat:
        new_cat = input(
            f"New category for '{person_id}' "
            f"[resident/visitor/watchlist] (current: {person.category}): "
        ).strip().lower()

    if not new_cat:
        print("Empty category; nothing changed.")
        return

    gallery.update_metadata(person_id, category=new_cat)
    gallery.save()
    print(f"Person '{person_id}' category updated to '{new_cat}'.")


# ---------------------------------------------------------------------------
# Reset-gallery command (Wave 2.1)
# ---------------------------------------------------------------------------


def cmd_reset_gallery(args: argparse.Namespace) -> None:
    """
    Danger operation: delete the encrypted face gallery file on disk.

    This is the explicit 'escape hatch' when the key changed or the file
    is corrupted. It does NOT recreate the gallery; a new one will be
    written automatically on the next successful enrollment/save.
    """
    _setup_logging()
    cfg = default_face_config()
    path_str = cfg.gallery.gallery_path
    if not path_str:
        print("Gallery path is not configured in face config.")
        return

    path = Path(path_str)

    if not path.exists():
        print(f"No gallery file found at: {path}")
        return

    if not args.force:
        print(f"About to DELETE gallery file:\n  {path}\n")
        print(
            "This will permanently remove all enrolled persons from this "
            "gallery. You cannot undo this operation."
        )
        confirm = input("Type 'DELETE' to confirm, or anything else to cancel: ").strip()
        if confirm != "DELETE":
            print("Reset cancelled.")
            return

    # Delete file
    path.unlink()
    logger.warning("Face gallery file %s deleted on user request.", path)
    print(f"Face gallery file deleted: {path}")
    print(
        "A new encrypted gallery will be created automatically the next time "
        "you enroll someone."
    )


# ---------------------------------------------------------------------------
# Main entrypoint / argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaitguard-enroll",
        description="GaitGuard Face Gallery enrollment CLI",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # enroll (classic 2D)
    p_enroll = sub.add_parser("enroll", help="Enroll a person via webcam (classic 2D)")
    p_enroll.add_argument("--name", type=str, default="", help="Person name (optional)")
    p_enroll.add_argument(
        "--category",
        type=str,
        default="",
        help="Person category [resident/visitor/watchlist] (default=resident)",
    )
    p_enroll.add_argument(
        "--condition",
        type=str,
        default="",
        help="Face condition label [neutral/glasses/cap/etc] (default=neutral)",
    )
    p_enroll.add_argument("--notes", type=str, default="", help="Optional notes")
    p_enroll.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Maximum number of face samples to capture (default=5)",
    )
    p_enroll.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default=0)",
    )
    p_enroll.add_argument(
        "--existing",
        type=str,
        default="",
        help=(
            "Existing person_id to append new templates to instead of creating "
            "a new person (for multi-condition enrollment)."
        ),
    )
    p_enroll.set_defaults(func=cmd_enroll)

    # guided-enroll (5-pose multi-view)
    p_guided = sub.add_parser(
        "guided-enroll",
        help="Guided 5-pose multi-view enrollment (3D-aware)",
    )
    p_guided.add_argument(
        "--person-id",
        type=str,
        default="",
        help="Optional explicit person_id (if omitted, a new p_XXXX is generated)",
    )
    p_guided.add_argument(
        "--name",
        type=str,
        default="",
        help="First name (optional, can also be prompted)",
    )
    p_guided.add_argument(
        "--surname",
        type=str,
        default="",
        help="Surname (optional, can also be prompted)",
    )
    p_guided.add_argument(
        "--category",
        type=str,
        default="",
        help="Person category [resident/visitor/watchlist] (default=resident)",
    )
    p_guided.add_argument(
        "--condition",
        type=str,
        default="",
        help="Face condition label [neutral/glasses/cap/etc] (default=neutral)",
    )
    p_guided.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes stored in template metadata",
    )
    p_guided.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default=0)",
    )
    p_guided.add_argument(
        "--frames-per-pose",
        type=int,
        default=20,
        help="Number of frames to capture per pose (default=20)",
    )
    p_guided.add_argument(
        "--min-quality",
        type=float,
        default=0.15,
        help="Minimum q_face for accepting a sample in guided mode (default=0.15)",
    )
    p_guided.add_argument(
        "--yaw-tolerance",
        type=float,
        default=25.0,
        help="Yaw tolerance in degrees around target pose (default=25)",
    )
    p_guided.add_argument(
        "--pitch-tolerance",
        type=float,
        default=25.0,
        help="Pitch tolerance in degrees around target pose (default=25)",
    )
    p_guided.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV enrollment window (headless mode)",
    )
    p_guided.set_defaults(func=cmd_guided_enroll)

    # NEW: enroll-watchlist (image-based)
    p_watch = sub.add_parser(
        "enroll-watchlist",
        help="Enroll a watchlist person using one or more external images",
    )
    p_watch.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the watchlist person (optional; will be prompted if empty)",
    )
    p_watch.add_argument(
        "--surname",
        type=str,
        default="",
        help="Surname of the watchlist person (optional)",
    )
    p_watch.add_argument(
        "--country",
        type=str,
        default="",
        help="Country / origin info (optional)",
    )
    p_watch.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text description (age, gender, case details, etc.)",
    )
    p_watch.add_argument(
        "--no-copy-images",
        action="store_true",
        help=(
            "Do NOT copy raw images into data/watchlist_raw; only embeddings "
            "and metadata will be stored."
        ),
    )
    p_watch.add_argument(
        "--raw-root",
        type=str,
        default="",
        help=(
            "Custom root directory for storing copies of watchlist images "
            "(default: data/watchlist_raw)."
        ),
    )
    p_watch.set_defaults(func=cmd_enroll_watchlist)

    # list
    p_list = sub.add_parser("list", help="List enrolled persons")
    p_list.set_defaults(func=cmd_list)

    # delete
    p_delete = sub.add_parser("delete", help="Delete a person from the gallery")
    p_delete.add_argument("person_id", nargs="?", help="person_id to delete")
    p_delete.set_defaults(func=cmd_delete)

    # rename
    p_rename = sub.add_parser("rename", help="Rename an existing person")
    p_rename.add_argument("person_id", help="person_id to rename")
    p_rename.add_argument(
        "name",
        nargs="?",
        help="New name (if omitted, you will be prompted)",
    )
    p_rename.set_defaults(func=cmd_rename)

    # set-category
    p_cat = sub.add_parser(
        "set-category",
        help="Update category [resident/visitor/watchlist] for an existing person",
    )
    p_cat.add_argument("person_id", help="person_id to update")
    p_cat.add_argument(
        "category",
        nargs="?",
        help="New category (if omitted, you will be prompted)",
    )
    p_cat.set_defaults(func=cmd_set_category)

    # reset-gallery
    p_reset = sub.add_parser(
        "reset-gallery",
        help="Delete the encrypted face gallery file (DANGEROUS).",
    )
    p_reset.add_argument(
        "--force",
        action="store_true",
        help="Skip interactive confirmation prompt.",
    )
    p_reset.set_defaults(func=cmd_reset_gallery)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return
    func(args)


if __name__ == "__main__":
    main()

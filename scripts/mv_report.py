
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from face.config import default_face_config
from face.multiview_types import (
    PoseBin,
    ALL_PRIMARY_BINS,
    MultiViewPersonModel,
    MultiViewBin,
    compute_coverage_score,
)
from identity.face_gallery import FaceGallery




def _format_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _posebin_order_key(pb: PoseBin) -> int:
    """
    Deterministic ordering for printing bins.
    Primary bins first, then OCCLUDED/UNKNOWN.
    """
    order = {
        PoseBin.FRONT: 0,
        PoseBin.LEFT: 1,
        PoseBin.RIGHT: 2,
        PoseBin.UP: 3,
        PoseBin.DOWN: 4,
        PoseBin.OCCLUDED: 5,
        PoseBin.UNKNOWN: 6,
    }
    return order.get(pb, 99)


def _summarise_angles(bin_obj: MultiViewBin) -> str:
    """
    Short summary of yaw/pitch ranges inside one bin.
    """
    if not bin_obj.samples:
        return "-"

    yaws = [s.yaw_deg for s in bin_obj.samples if s.yaw_deg is not None]
    pitches = [s.pitch_deg for s in bin_obj.samples if s.pitch_deg is not None]

    parts = []
    if yaws:
        parts.append(
            f"yaw≈{_format_float(float(np.mean(yaws)), 1)} "
            f"[{_format_float(float(np.min(yaws)), 1)}, "
            f"{_format_float(float(np.max(yaws)), 1)}]"
        )
    if pitches:
        parts.append(
            f"pitch≈{_format_float(float(np.mean(pitches)), 1)} "
            f"[{_format_float(float(np.min(pitches)), 1)}, "
            f"{_format_float(float(np.max(pitches)), 1)}]"
        )

    return "; ".join(parts) if parts else "-"


def _print_person_report(
    person_id: str,
    model: MultiViewPersonModel,
    *,
    gallery: FaceGallery,
) -> None:
    """
    Detailed per-person report: bins, coverage, quality, pose ranges.
    """
    person = gallery.persons.get(person_id)

    print("=" * 72)
    print(f"Person: {person_id}")
    if person is not None:
        name = person.name or "(none)"
        print(f"  Name      : {name}")
        print(f"  Category  : {person.category}")
        print(f"  Templates : {len(person.templates)}")
    print(f"  Coverage  : {model.num_populated_bins()}/{len(ALL_PRIMARY_BINS)} "
          f"({model.coverage_score() * 100:.1f}%)")
    print(f"  Created   : {model.created_at:.0f}")
    print(f"  Updated   : {model.updated_at:.0f}")
    print()

    print(f"{'Bin':<10} {'#samples':>9} {'avg_q':>8} {'angles':<40}")
    print("-" * 72)

    printed_bins = set()

    def _print_bin_row(pb: PoseBin, bin_obj: Optional[MultiViewBin]) -> None:
        if bin_obj is None or not bin_obj.samples:
            print(f"{pb.value:<10} {0:>9} {'-':>8} {'-':<40}")
            return
        num = len(bin_obj.samples)
        avg_q = _format_float(bin_obj.avg_quality, 3)
        angles = _summarise_angles(bin_obj)
        print(f"{pb.value:<10} {num:>9} {avg_q:>8} {angles:<40}")

    for pb in ALL_PRIMARY_BINS:
        bin_obj = model.bins.get(pb)
        _print_bin_row(pb, bin_obj)
        printed_bins.add(pb)

    extra_bins = sorted(
        [pb for pb in model.bins.keys() if pb not in printed_bins],
        key=_posebin_order_key,
    )
    if extra_bins:
        print("-" * 72)
        for pb in extra_bins:
            bin_obj = model.bins.get(pb)
            _print_bin_row(pb, bin_obj)

    print("=" * 72)
    print()


def _print_global_summary(models: Dict[str, MultiViewPersonModel]) -> None:
    """
    Global coverage summary over all persons.
    """
    print("=" * 72)
    print("Global Multi-View Coverage Summary")
    print("=" * 72)

    num_persons = len(models)
    print(f"Persons with any templates : {num_persons}")
    if num_persons == 0:
        print("No multi-view models found (empty gallery or no templates).")
        print("=" * 72)
        return

    avg_cov = compute_coverage_score(models)
    fully_covered = sum(
        1
        for m in models.values()
        if m.num_populated_bins() == len(ALL_PRIMARY_BINS)
    )

    print(f"Average coverage           : {avg_cov * 100:.1f}%")
    print(f"Fully covered (all bins)   : {fully_covered}/{num_persons}")
    print()

    hist = {}
    for m in models.values():
        n = m.num_populated_bins()
        hist[n] = hist.get(n, 0) + 1

    print("Histogram: #bins_populated -> #persons")
    for n in sorted(hist.keys()):
        print(f"  {n:2d} bins : {hist[n]:3d} person(s)")

    print("=" * 72)
    print()




def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.mv_report",
        description=(
            "Inspect multi-view (pseudo-3D) face models built from the "
            "encrypted face gallery.\n\n"
            "Usage:\n"
            "  python -m scripts.mv_report            # global summary\n"
            "  python -m scripts.mv_report p_0007     # detailed report for one person\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "person_id",
        nargs="?",
        default=None,
        help="Optional person_id (e.g. 'p_0007') to show a detailed per-person report.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("scripts.mv_report")

    parser = build_arg_parser()
    args = parser.parse_args(argv)


    base_dir = Path(__file__).resolve().parents[1]

    face_cfg = default_face_config(
        prefer_gpu=True,
        base_dir=base_dir,
        face_section=None,
    )

    log.info("Face gallery path: %s", face_cfg.gallery.gallery_path)

    gallery = FaceGallery(face_cfg.gallery)

    models = gallery.get_all_multiview_models()

    if not models:
        print("No multi-view models available. "
              "Ensure you have enrolled people (ideally via guided 5-pose).")
        return

    if args.person_id:
        pid = args.person_id.strip()
        mv = models.get(pid)
        if mv is None:
            print(f"No MultiViewPersonModel found for person_id '{pid}'.")
            print("Tip: run without arguments to see global coverage.")
            return

        _print_person_report(pid, mv, gallery=gallery)
        _print_global_summary(models)
        return

    _print_global_summary(models)


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np

from face.multiview_types import PoseBin
from face.multiview_gallery_view import MultiViewGalleryView
from typing import Any



logger = logging.getLogger(__name__)



@dataclass
class PersonBinReport:
    pose_bin: PoseBin
    num_samples: int
    centroid_quality: float
    avg_quality: float
    yaw_range: str
    pitch_range: str


@dataclass
class PersonMultiViewReport:
    person_id: str
    name: Optional[str]
    total_samples: int
    bins_present: int
    completeness_score: float
    bins: List[PersonBinReport]


@dataclass
class GlobalMultiViewStats:
    total_persons: int
    persons_with_all_bins: int
    persons_missing_bins: int
    avg_bins_present: float
    missing_counts: Dict[str, int]



class MultiViewDiagnostics:
    """
    Read-only diagnostic interface for multi-view models.
    Works with an already built MultiViewGalleryView.

    NOTE:
      - This does not modify gallery, multiview models, or any runtime state.
      - Safe to call from CLI tools, notebooks, or professor demos.
    """

    def __init__(self, mv_gallery: MultiViewGalleryView):
        self._mv = mv_gallery


    def report_person(self, person_id: str) -> Optional[PersonMultiViewReport]:
        """
        Produce a structured diagnostic report for a single person.

        Returns
        -------
        PersonMultiViewReport or None
            None if the person has no multi-view model (no templates
            or not yet built), otherwise a rich summary of pose bins.
        """
        model = self._mv.models.get(person_id)
        if model is None:
            logger.warning(f"[diagnostics] No multiview model for '{person_id}'")
            return None

        bins: List[PersonBinReport] = []
        total_samples = 0
        bins_present = 0

        for b in PoseBin:
            bin_data = model.bins.get(b)
            if bin_data is None:
                continue

            bins_present += 1
            total_samples += len(bin_data.samples)

            yaw_vals = [s.yaw for s in bin_data.samples]
            pitch_vals = [s.pitch for s in bin_data.samples]
            qual_vals = [s.quality for s in bin_data.samples]

            yaw_range = f"{min(yaw_vals):.1f} → {max(yaw_vals):.1f}" if yaw_vals else "N/A"
            pitch_range = f"{min(pitch_vals):.1f} → {max(pitch_vals):.1f}" if pitch_vals else "N/A"
            avg_quality = float(np.mean(qual_vals)) if qual_vals else 0.0

            bins.append(
                PersonBinReport(
                    pose_bin=b,
                    num_samples=len(bin_data.samples),
                    centroid_quality=bin_data.centroid_quality,
                    avg_quality=avg_quality,
                    yaw_range=yaw_range,
                    pitch_range=pitch_range,
                )
            )

        completeness = model.completeness_score

        return PersonMultiViewReport(
            person_id=person_id,
            name=model.name,
            total_samples=total_samples,
            bins_present=bins_present,
            completeness_score=completeness,
            bins=bins,
        )

    def missing_bins_for_person(self, person_id: str) -> List[str]:
        """
        Return a list of pose-bin names that are missing for this person.

        This is purely diagnostic and does not modify any state.
        """
        model = self._mv.models.get(person_id)
        if model is None:
            return [b.name for b in PoseBin]
        return [b.name for b in PoseBin if b not in model.bins]


    def global_summary(self) -> GlobalMultiViewStats:
        """
        Compute gallery-wide statistics:
            - how many persons
            - how many have all bins
            - which bins are commonly missing
        """
        total = len(self._mv.models)
        if total == 0:
            return GlobalMultiViewStats(
                total_persons=0,
                persons_with_all_bins=0,
                persons_missing_bins=0,
                avg_bins_present=0.0,
                missing_counts={b.name: 0 for b in PoseBin},
            )

        full = 0
        missing = 0
        total_bins = 0

        missing_counts = {b.name: 0 for b in PoseBin}

        for pid, model in self._mv.models.items():
            present = [b for b in PoseBin if b in model.bins]

            total_bins += len(present)
            if len(present) == len(PoseBin):
                full += 1
            else:
                missing += 1
                for b in PoseBin:
                    if b not in model.bins:
                        missing_counts[b.name] += 1

        avg_bins = total_bins / total

        return GlobalMultiViewStats(
            total_persons=total,
            persons_with_all_bins=full,
            persons_missing_bins=missing,
            avg_bins_present=avg_bins,
            missing_counts=missing_counts,
        )


    @staticmethod
    def _grade_completeness(score: float) -> str:
        """
        Map a completeness_score in [0, 1] to a human label.

        This does not affect any numeric logic; used only in ASCII renderers.
        """
        if score >= 0.9:
            return "EXCELLENT"
        if score >= 0.75:
            return "GOOD"
        if score >= 0.5:
            return "FAIR"
        return "POOR"


    def render_person_ascii(self, person_id: str) -> str:
        """
        Render a human-readable ASCII report for a person.

        Includes:
          - total samples
          - bins present / total
          - completeness score + grade
          - per-bin sample counts, qualities, and yaw/pitch ranges
        """
        rep = self.report_person(person_id)
        if rep is None:
            return f"No multiview model for {person_id}"

        grade = self._grade_completeness(rep.completeness_score)

        lines: List[str] = []
        lines.append(f"=== Multi-View Report: {rep.person_id} ({rep.name}) ===")
        lines.append(f"Total samples: {rep.total_samples}")
        lines.append(f"Bins present: {rep.bins_present}/{len(PoseBin)}")
        lines.append(f"Completeness score: {rep.completeness_score:.3f} [{grade}]")
        lines.append("")

        for b in rep.bins:
            lines.append(f"[{b.pose_bin.name}] samples={b.num_samples}")
            lines.append(f"   centroid_quality={b.centroid_quality:.3f}")
            lines.append(f"   avg_quality={b.avg_quality:.3f}")
            lines.append(f"   yaw_range={b.yaw_range}")
            lines.append(f"   pitch_range={b.pitch_range}")
            lines.append("")

        missing_bins = self.missing_bins_for_person(person_id)
        if missing_bins:
            lines.append("Missing bins: " + ", ".join(missing_bins))

        return "\n".join(lines)

    def render_person_compact_ascii(self, person_id: str) -> str:
        """
        Render a compact one-line summary for quick overviews.

        Example:
          p_0001 (Alice): bins=3/5, completeness=0.78 [GOOD]
        """
        rep = self.report_person(person_id)
        if rep is None:
            return f"{person_id}: no multiview model"

        grade = self._grade_completeness(rep.completeness_score)
        return (
            f"{rep.person_id} ({rep.name}): "
            f"bins={rep.bins_present}/{len(PoseBin)}, "
            f"completeness={rep.completeness_score:.2f} [{grade}]"
        )

    def render_global_ascii(self) -> str:
        """
        Render a global summary report for the entire gallery.

        Shows:
          - total persons with multi-view models
          - how many have full bin coverage
          - average number of bins present
          - how often each bin is missing
        """
        stat = self.global_summary()

        lines: List[str] = []
        lines.append("=== Multi-View Global Summary ===")
        lines.append(f"Total persons: {stat.total_persons}")
        lines.append(f"Persons with ALL bins: {stat.persons_with_all_bins}")
        lines.append(f"Persons missing bins: {stat.persons_missing_bins}")
        lines.append(f"Avg bins present: {stat.avg_bins_present:.2f}")
        lines.append("")
        lines.append("Missing bins counts:")
        for b, cnt in stat.missing_counts.items():
            lines.append(f"   {b:>6}: {cnt}")

        return "\n".join(lines)

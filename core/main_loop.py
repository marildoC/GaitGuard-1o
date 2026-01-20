from __future__ import annotations

import time
import logging

import cv2
import numpy as np
from dataclasses import asdict
from core.interfaces import IdentityEngine as IdentityEngineBase

from schemas import Frame
from .camera import CameraSource
from .dummies import (
    DummyEventsEngine,
    DummyAlertEngine,
)
from .config import load_config
from .logging_setup import setup_logging
from .device import select_device, get_gpu_memory_stats
from ui.overlay import draw_overlay

from perception.perception_engine import Phase1PerceptionEngine

from identity.identity_engine import FaceIdentityEngine

from .metrics import FaceMetrics

from .governance_metrics import MetricsCollector

try:
    from identity.identity_engine_multiview import IdentityEngineMultiView
except Exception:
    IdentityEngineMultiView = None

try:
    from source_auth.config import default_source_auth_config
    from source_auth.engine import SourceAuthEngine
    from source_auth.diagnostics import format_source_auth_reason
except Exception:
    default_source_auth_config = None
    SourceAuthEngine = None
    format_source_auth_reason = None


def run() -> None:
    """
    Run the GaitGuard pipeline (Phase-1 + Phase-2A Face).

    Pipeline:
        Frame (camera) ->
        Perception (detect + track + ring buffers) ->
        Identity (classic 2D gallery OR multiview OR hybrid) ->
        SourceAuth (real head vs phone/screen/photo) ->
        Events (dummy) ->
        Alerts (dummy) ->
        UI overlay (+ Wave-3 telemetry)

    Press ESC in the window to exit.
    """
    cfg = load_config()
    setup_logging(cfg.paths.logs_dir)
    log = logging.getLogger("gaitguard.main")

    device, use_half = select_device(prefer_gpu=cfg.runtime.use_gpu)
    log.info("Runtime device=%s | half=%s", device, use_half)

    perception = Phase1PerceptionEngine()

    identity_mode_requested: str = "classic"
    identity_mode_source: str = "default"

    face_cfg = getattr(cfg, "face", None)
    raw_mode = None
    if face_cfg is not None and hasattr(face_cfg, "identity_mode"):
        raw_mode = getattr(face_cfg, "identity_mode", None)
        if isinstance(raw_mode, str) and raw_mode.strip():
            identity_mode_requested = raw_mode.strip().lower()
            identity_mode_source = "face.identity_mode"

    if identity_mode_source == "default":
        identity_cfg = getattr(cfg, "identity", None)
        raw_mode = None
        if identity_cfg is not None:
            if hasattr(identity_cfg, "mode"):
                raw_mode = getattr(identity_cfg, "mode")
            elif isinstance(identity_cfg, dict):
                raw_mode = identity_cfg.get("mode")

        if isinstance(raw_mode, str) and raw_mode.strip():
            identity_mode_requested = raw_mode.strip().lower()
            identity_mode_source = "identity.mode"

    if identity_mode_source == "default":
        use_mv_flag = bool(getattr(cfg.runtime, "use_multiview_engine", False))
        if use_mv_flag:
            identity_mode_requested = "multiview"
            identity_mode_source = "runtime.use_multiview_engine"

    if identity_mode_requested not in ("classic", "multiview", "hybrid"):
        log.warning(
            "Invalid identity_mode '%s' (source=%s); falling back to 'classic'. "
            "Allowed: classic, multiview, hybrid.",
            identity_mode_requested,
            identity_mode_source,
        )
        identity_mode_requested = "classic"

    multiview_available = IdentityEngineMultiView is not None

    identity_primary: IdentityEngineBase
    identity_secondary: IdentityEngineBase | None = None
    identity_mode_active: str

    if identity_mode_requested == "multiview" and multiview_available:
        identity_primary = IdentityEngineMultiView()
        identity_mode_active = "multiview"
        log.info("Identity engine: multiview only (pseudo-3D, Wave-3)")
    elif identity_mode_requested == "hybrid" and multiview_available:
        identity_primary = IdentityEngineMultiView()
        identity_secondary = FaceIdentityEngine()
        identity_mode_active = "hybrid"
        log.info(
            "Identity engine: hybrid (primary=multiview, secondary=classic for comparison)"
        )
    else:
        if identity_mode_requested in ("multiview", "hybrid") and not multiview_available:
            log.warning(
                "Multiview requested (mode=%s, source=%s) but "
                "IdentityEngineMultiView is not available. "
                "Falling back to classic FaceIdentityEngine.",
                identity_mode_requested,
                identity_mode_source,
            )
        identity_primary = FaceIdentityEngine()
        identity_mode_active = "classic"
        log.info("Identity engine: classic only (Phase-2A face route)")

    log.info(
        "Identity selection: requested=%s (source=%s, runtime.use_multiview_engine=%s), "
        "active=%s",
        identity_mode_requested,
        identity_mode_source,
        getattr(cfg.runtime, "use_multiview_engine", None),
        identity_mode_active,
    )

    if identity_mode_active == "multiview":
        window_title = "GaitGuard 1.0 - Phase 1 + Face 2A (3D multiview)"
    elif identity_mode_active == "hybrid":
        window_title = "GaitGuard 1.0 - Phase 1 + Face 2A (hybrid classic+3D)"
    else:
        window_title = "GaitGuard 1.0 - Phase 1 + Face 2A"

    events_engine = DummyEventsEngine()
    alert_engine = DummyAlertEngine()

    metrics: FaceMetrics | None = None
    log_face_metrics = bool(getattr(cfg.runtime, "log_face_metrics", False))
    if log_face_metrics:
        window_sec = float(getattr(cfg.runtime, "metrics_window_sec", 5.0))
        metrics = FaceMetrics(window_sec=window_sec, log_every_sec=5.0)
        log.info(
            "FaceMetrics telemetry enabled (window=%.1fs, log_every=5s)",
            window_sec,
        )

    source_auth_engine = None
    source_auth_enabled = True
    try:
        if hasattr(cfg, 'governance') and hasattr(cfg.governance, 'source_auth'):
            sa_config = cfg.governance.source_auth
            source_auth_enabled = bool(getattr(sa_config, 'enabled', True))
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning(f"Could not read source_auth config: {e}; defaulting to enabled")
        source_auth_enabled = True
    
    if source_auth_enabled and SourceAuthEngine is not None and default_source_auth_config is not None:
        try:
            sa_cfg = default_source_auth_config(face_cfg=face_cfg)
            source_auth_engine = SourceAuthEngine(sa_cfg)
            log.info(
                "SourceAuth engine initialised "
                "(motion + screen + background cues; annotating IdentityDecision)."
            )
        except Exception:
            log.exception(
                "Failed to initialise SourceAuth engine; "
                "continuing without source authenticity checks."
            )
            source_auth_engine = None
    elif not source_auth_enabled:
        log.info("SourceAuth engine disabled via governance.source_auth.enabled=false")

    scheduler = None
    scheduler_enabled = False
    if hasattr(cfg, "governance") and hasattr(cfg.governance, "scheduler"):
        try:
            from core.scheduler import create_scheduler_from_config
            scheduler_cfg_dict = cfg.governance.scheduler
            if hasattr(scheduler_cfg_dict, "enabled"):
                scheduler_cfg_dict = asdict(scheduler_cfg_dict)
            
            if isinstance(scheduler_cfg_dict, dict):
                scheduler = create_scheduler_from_config(scheduler_cfg_dict)
                scheduler_enabled = scheduler_cfg_dict.get("enabled", True)
                log.info(
                    "Phase D Scheduler initialised "
                    "(budget_policy=%s, enabled=%s)",
                    scheduler_cfg_dict.get("budget_policy", "adaptive"),
                    scheduler_enabled,
                )
            else:
                log.warning(f"governance.scheduler config is {type(scheduler_cfg_dict)}, expected dict or dataclass; scheduler disabled")
        except Exception:
            log.exception("Failed to initialise Phase D scheduler; continuing without scheduler")
            scheduler = None
            scheduler_enabled = False

    merge_manager = None
    merge_enabled = False
    if hasattr(cfg, "governance") and hasattr(cfg.governance, "merge"):
        try:
            from identity.merge_manager import create_merge_config_from_dict, MergeManager
            merge_cfg_dict = cfg.governance.merge
            if hasattr(merge_cfg_dict, "enabled"):
                merge_cfg_dict = asdict(merge_cfg_dict)
            
            if isinstance(merge_cfg_dict, dict):
                merge_config = create_merge_config_from_dict(merge_cfg_dict)
                merge_manager = MergeManager(merge_config)
                merge_enabled = merge_cfg_dict.get("enabled", True)
                log.info(
                    "Phase E Merge Manager initialised "
                    "(strategy=%s, enabled=%s)",
                    merge_cfg_dict.get("merge_strategy", {}).get("mode", "conservative"),
                    merge_enabled,
                )
            else:
                log.warning(f"governance.merge config is {type(merge_cfg_dict)}, expected dict or dataclass; merge manager disabled")
        except Exception:
            log.exception("Failed to initialise Phase E merge manager; continuing without merge manager")
            merge_manager = None
            merge_enabled = False

    if hasattr(perception, "warmup"):
        try:
            log.info("Warming up perception engine (if supported)...")
            perception.warmup()
        except Exception:
            log.exception("Perception warmup failed")

    if hasattr(identity_primary, "warmup"):
        try:
            log.info("Warming up primary identity engine (if supported)...")
            identity_primary.warmup()
        except Exception:
            log.exception("Primary identity warmup failed")

    if identity_secondary is not None and hasattr(identity_secondary, "warmup"):
        try:
            log.info("Warming up secondary identity engine (hybrid mode)...")
            identity_secondary.warmup()
        except Exception:
            log.exception("Secondary identity warmup failed")

    if source_auth_engine is not None and hasattr(source_auth_engine, "warmup"):
        try:
            log.info("Warming up SourceAuth engine (if supported)...")
            source_auth_engine.warmup()
        except Exception:
            log.exception("SourceAuth warmup failed")

    src = CameraSource(
        cam_index=cfg.camera.index,
        w=cfg.camera.width,
        h=cfg.camera.height,
        fps=cfg.camera.fps,
        buffersize=1,
    )
    src.start()

    log.info(
        "GaitGuard pipeline started "
        "(Phase-1 + Face identity, mode=%s). Press ESC to exit.",
        identity_mode_active,
    )

    frame_id = 0
    camera_id = "cam0"

    t0 = time.perf_counter()
    frames = 0
    last_fps = 0.0
    prev_frame_ts = -1.0

    hybrid_agree = 0
    hybrid_total = 0
    
    governance_enabled = bool(getattr(cfg, "governance", {}).enabled if hasattr(cfg, "governance") else False)
    metrics_collector = None
    if governance_enabled:
        from core.governance_metrics import get_metrics_collector
        metrics_interval = float(getattr(cfg.governance.debug, "emit_metrics_every_sec", 1.0))
        metrics_collector = get_metrics_collector()
        metrics_collector.interval_sec = metrics_interval
        log.info("Governance metrics collection enabled (emit every %.1f sec)", metrics_interval)

    try:
        while True:
            img = src.read_latest(timeout=1.0)
            if img is None:
                continue

            ts = time.perf_counter()
            h, w = img.shape[:2]

            actual_fps = last_fps if last_fps > 0 else 30.0
            if prev_frame_ts > 0:
                frame_time = ts - prev_frame_ts
                if frame_time > 0:
                    actual_fps = 1.0 / frame_time
            prev_frame_ts = ts



            frame = Frame(
                frame_id=frame_id,
                ts=ts,
                camera_id=camera_id,
                size=(w, h),
                image=img,
            )
            frame_id += 1

            tracks = perception.process_frame(frame)

            schedule_context = None
            if scheduler_enabled and scheduler is not None:
                try:
                    binding_states = {}
                    if identity_secondary is None:
                        binding_states = getattr(identity_primary, "get_binding_states", lambda: {})()
                    else:
                        binding_states = getattr(identity_primary, "get_binding_states", lambda: {})()
                    
                    schedule_context = scheduler.compute_schedule(
                        track_ids=list(t.track_id for t in tracks) if tracks else [],
                        binding_states=binding_states,
                        current_ts=ts,
                        actual_fps=actual_fps,
                    )
                except Exception:
                    log.exception("Failed to compute scheduler context; continuing without scheduling")
                    schedule_context = None

            signals_active = None

            if identity_secondary is None:
                try:
                    signals = identity_primary.update_signals(frame, tracks, schedule_context=schedule_context)
                except TypeError:
                    signals = identity_primary.update_signals(frame, tracks)
                
                decisions = identity_primary.decide(signals)
                signals_active = signals
            else:
                try:
                    signals_primary = identity_primary.update_signals(frame, tracks, schedule_context=schedule_context)
                except TypeError:
                    signals_primary = identity_primary.update_signals(frame, tracks)
                decisions_primary = identity_primary.decide(signals_primary)

                try:
                    signals_secondary = identity_secondary.update_signals(frame, tracks, schedule_context=schedule_context)
                except TypeError:
                    signals_secondary = identity_secondary.update_signals(frame, tracks)
                decisions_secondary = identity_secondary.decide(signals_secondary)

                decisions = decisions_primary
                signals_active = signals_primary

                try:
                    primary_map = {
                        d.track_id: (d.identity_id, getattr(d, "category", None))
                        for d in decisions_primary
                    }
                    secondary_map = {
                        d.track_id: (d.identity_id, getattr(d, "category", None))
                        for d in decisions_secondary
                    }
                    common_ids = set(primary_map.keys()) & set(secondary_map.keys())
                    if common_ids:
                        for tid in common_ids:
                            if primary_map[tid] == secondary_map[tid]:
                                hybrid_agree += 1
                            hybrid_total += 1
                        if hybrid_total > 0 and (frame_id % 60 == 0):
                            agree_ratio = hybrid_agree / max(hybrid_total, 1)
                            log.info(
                                "Hybrid diagnostics: agree=%d / %d (%.1f%%)",
                                hybrid_agree,
                                hybrid_total,
                                agree_ratio * 100.0,
                            )
                except Exception:
                    log.exception("Hybrid diagnostics failed")

            if source_auth_engine is not None and signals_active is not None:
                try:
                    sa_results = source_auth_engine.update(
                        frame,
                        tracks,
                        signals_active,
                    )

                    if sa_results:
                        for dec in decisions:
                            tid = dec.track_id
                            sa = sa_results.get(tid)

                            if sa is None:
                                continue

                            try:
                                sa_score = float(getattr(sa, "source_auth_score", 0.0))
                            except Exception:
                                sa_score = 0.0

                            try:
                                dec.source_auth_score = sa_score
                            except Exception:
                                pass

                            try:
                                state = getattr(sa, "state", None)
                                sa_state = str(state) if state is not None else "UNCERTAIN"
                            except Exception:
                                sa_state = "UNCERTAIN"

                            try:
                                dec.source_auth_state = sa_state
                            except Exception:
                                pass

                            reason_fragment = None
                            if format_source_auth_reason is not None:
                                try:
                                    reason_fragment = format_source_auth_reason(sa, debug=None)
                                except Exception:
                                    reason_fragment = None

                            if reason_fragment:
                                try:
                                    existing_reason = getattr(dec, "reason", "") or ""
                                    if existing_reason:
                                        dec.reason = f"{existing_reason}|{reason_fragment}"
                                    else:
                                        dec.reason = reason_fragment
                                except Exception:
                                    pass

                            if hasattr(dec, "risk_label"):
                                try:
                                    identity_score = getattr(dec, "score", None)
                                    if identity_score is None:
                                        identity_score = getattr(dec, "confidence", None)

                                    strong_identity = False
                                    if identity_score is not None:
                                        try:
                                            strong_identity = float(identity_score) >= 0.80
                                        except Exception:
                                            strong_identity = False

                                    sa_state_upper = sa_state.upper()
                                    sa_real = sa_state_upper in ("REAL", "LIKELY_REAL")
                                    sa_spoof = sa_state_upper in ("SPOOF", "LIKELY_SPOOF")

                                    if strong_identity and sa_real:
                                        dec.risk_label = "ID_STRONG_SA_REAL"
                                    elif strong_identity and sa_spoof:
                                        dec.risk_label = "ID_STRONG_SA_SPOOF"
                                    elif (not strong_identity) and sa_real:
                                        dec.risk_label = "ID_WEAK_SA_REAL"
                                    elif (not strong_identity) and sa_spoof:
                                        dec.risk_label = "ID_WEAK_SA_SPOOF"
                                    else:
                                        dec.risk_label = "ID_SA_UNCERTAIN"
                                except Exception:
                                    pass

                except Exception:
                    log.exception(
                        "SourceAuth update failed; "
                        "continuing without source authenticity for this frame."
                    )

            if merge_manager is not None and merge_enabled:
                try:
                    merge_manager.current_time = ts
                    
                    for track in tracks:
                        track_id = track.track_id
                        
                        binding_state = None
                        for dec in decisions:
                            if dec.track_id == track_id:
                                binding_state = {
                                    'person_id': getattr(dec, 'identity_id', None),
                                    'status': getattr(dec, 'binding_state', 'UNKNOWN'),
                                    'confidence': getattr(dec, 'score', getattr(dec, 'confidence', 0.0))
                                }
                                break
                        
                        appearance_features = None
                        if hasattr(track, 'embedding') and track.embedding is not None:
                            appearance_features = track.embedding
                        elif hasattr(track, 'appearance_features') and track.appearance_features is not None:
                            appearance_features = track.appearance_features
                        
                        quality_samples = 0
                        if hasattr(track, 'face_hits'):
                            quality_samples = track.face_hits
                        elif hasattr(track, 'quality_samples'):
                            quality_samples = track.quality_samples
                        
                        merge_manager.on_tracklet_updated(
                            tracklet_id=track_id,
                            binding_state=binding_state,
                            appearance_features=appearance_features,
                            current_position=(float(track.x), float(track.y)) if hasattr(track, 'x') else (0.0, 0.0),
                            track_length=getattr(track, 'frame_hits', 0),
                            quality_samples=quality_samples,
                            timestamp=ts
                        )
                    
                    if frame_id % 10 == 0:
                        active_tracklets_dict = {}
                        for track in tracks:
                            from identity.merge_manager import MergeCandidate
                            candidate = MergeCandidate(
                                tracklet_id=track.track_id,
                                start_time=ts - (getattr(track, 'frame_hits', 0) / max(last_fps, 1.0)),
                                track_length=getattr(track, 'frame_hits', 0),
                                first_position=np.array([getattr(track, 'x', 0.0), getattr(track, 'y', 0.0)], dtype=np.float32),
                                quality_samples=quality_samples
                            )
                            active_tracklets_dict[track.track_id] = candidate
                        
                        merges_executed = merge_manager.check_and_execute_merges(active_tracklets_dict)
                        
                        if merges_executed > 0 and (frame_id % 30 == 0):
                            metrics = merge_manager.get_metrics()
                            log.debug(
                                "Phase E Merge: %d executions this frame, "
                                "total=%d merges, reversals=%d, avg_score=%.1f",
                                merges_executed,
                                metrics.merges_executed,
                                metrics.merge_reversals,
                                metrics.average_merge_score
                            )
                    
                    if frame_id % 100 == 0:
                        merge_manager.cleanup_old_tracklets()
                
                except Exception:
                    log.exception("Phase E merge manager update failed; continuing without merge manager")

            events = events_engine.update(frame, tracks, decisions)
            alerts = alert_engine.update(frame, events, decisions)

            if metrics is not None:
                try:
                    metrics.update(decisions, tracks, ts)
                    metrics.maybe_log(ts)
                except Exception:
                    log.exception("FaceMetrics update/maybe_log failed")
            
            if metrics_collector is not None:
                try:
                    metrics_collector.metrics.fps_estimate = last_fps
                    metrics_collector.metrics.track_count = len(tracks)
                    
                    binding_states = {}
                    for dec in decisions:
                        state = getattr(dec, "binding_state", "UNKNOWN")
                        binding_states[state] = binding_states.get(state, 0) + 1
                    metrics_collector.metrics.binding_state_counts = binding_states
                    
                    if len(tracks) > 0:
                        unknown_cnt = binding_states.get("UNKNOWN", 0)
                        pending_cnt = binding_states.get("PENDING", 0)
                        confirmed_cnt = len(tracks) - unknown_cnt - pending_cnt
                        metrics_collector.metrics.unknown_rate = unknown_cnt / len(tracks)
                        metrics_collector.metrics.pending_rate = pending_cnt / len(tracks)
                        metrics_collector.metrics.confirmed_rate = max(0.0, confirmed_cnt / len(tracks))
                    
                    metrics_collector.maybe_emit()
                except Exception:
                    log.exception("Governance metrics collection failed")

            frames += 1
            if frames % 30 == 0:
                elapsed = ts - t0
                last_fps = frames / max(elapsed, 1e-6)
                
                gpu_stats = get_gpu_memory_stats()
                gpu_info = ""
                if gpu_stats["status"] != "CPU_MODE":
                    gpu_info = f" | GPU: {gpu_stats['allocated_gb']:.2f}GB/{gpu_stats['total_gb']:.2f}GB ({gpu_stats['percent_used']:.1f}%)"
                    if gpu_stats["status"] == "CRITICAL":
                        log.warning(
                            "🚨 GPU CRITICAL: FPS=%.1f | tracks=%d | alerts=%d%s",
                            last_fps,
                            len(tracks),
                            len(alerts),
                            gpu_info,
                        )
                    elif gpu_stats["status"] == "WARNING":
                        log.warning(
                            "⚠️  GPU WARNING: FPS=%.1f | tracks=%d | alerts=%d%s",
                            last_fps,
                            len(tracks),
                            len(alerts),
                            gpu_info,
                        )
                    else:
                        log.info(
                            "FPS=%.1f | tracks=%d | alerts=%d%s",
                            last_fps,
                            len(tracks),
                            len(alerts),
                            gpu_info,
                        )
                else:
                    log.info(
                        "FPS=%.1f | tracks=%d | alerts=%d",
                        last_fps,
                        len(tracks),
                        len(alerts),
                    )

            try:
                display_img = draw_overlay(
                    frame,
                    tracks,
                    decisions,
                    events,
                    alerts,
                    ui_cfg=getattr(cfg, "ui", None),
                    fps=last_fps,
                )
            except TypeError:
                display_img = draw_overlay(frame, tracks, decisions, events, alerts)

            cv2.imshow(window_title, display_img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        try:
            if src is not None and hasattr(src, "stop"):
                src.stop()
        except Exception:
            log.exception("Error while stopping camera source")
        cv2.destroyAllWindows()
        log.info("GaitGuard pipeline stopped.")


if __name__ == "__main__":
    run()

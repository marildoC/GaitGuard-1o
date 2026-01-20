"""
GaitGuard Gait Subsystem

This package contains the gait recognition components:
- gait/config.py: Configuration management
- gait/gait_extractor.py: Pose sequence → embedding model
- gait/gait_gallery.py: FAISS-based identity matching
- gait/gait_engine.py: Main orchestrator

The gait subsystem is designed to work standalone for testing
and integrate with the main GaitGuard face system via cybernetic fusion.
"""

__version__ = "1.0.0"

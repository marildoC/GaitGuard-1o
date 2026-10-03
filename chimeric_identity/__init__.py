
__version__ = "2.0.0"
__author__ = "GaitGuard Team"


from chimeric_identity.simple_fusion import (
    SimpleFusionEngine,
    FusionResult,
    FusionState,
    FusionWeights,
    FaceInput,
    GaitInput,
    get_fusion_engine,
    format_fusion_result,
    create_face_input_from_decision,
    create_gait_input_from_signal,
)

from chimeric_identity.identity_registry import (
    IdentityRegistry,
    PersonRecord,
    get_identity_registry,
)

from chimeric_identity.simple_runner import (
    SimpleChimericRunner,
    SimpleRunnerConfig,
    run_chimeric,
)

from chimeric_identity.enrollment_integration import (
    ChimericEnrollment,
    sync_existing_galleries,
)


__all__ = [
    "__version__",
    
    "SimpleFusionEngine",
    "FusionResult",
    "FusionState", 
    "FusionWeights",
    "FaceInput",
    "GaitInput",
    "get_fusion_engine",
    "format_fusion_result",
    "create_face_input_from_decision",
    "create_gait_input_from_signal",
    
    "IdentityRegistry",
    "PersonRecord",
    "get_identity_registry",
    
    "SimpleChimericRunner",
    "SimpleRunnerConfig",
    "run_chimeric",
    
    "ChimericEnrollment",
    "sync_existing_galleries",
]

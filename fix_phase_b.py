"""Fix Phase B test file - replace enum assertions with tuple handling"""

import re

with open('tests/test_phase_b_evidence_gate.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

patterns = [
    ("decision = gate.decide(evidence)\n        \n        assert decision in [GateDecision.REJECT, GateDecision.HOLD], \\\n            f\"Very dark face should be REJECT or HOLD, got {decision}\"",
     "decision = gate.decide(evidence)\n        decision_status = decision[0] if isinstance(decision, tuple) else decision\n        \n        assert decision_status in ['REJECT', 'HOLD', GateDecision.REJECT, GateDecision.HOLD], \\\n            f\"Very dark face should be REJECT or HOLD, got {decision}\""),
     
    ("decision = gate.decide(evidence)\n        \n        assert decision in [GateDecision.REJECT, GateDecision.HOLD], \\\n            f\"Very bright face should be REJECT or HOLD, got {decision}\"",
     "decision = gate.decide(evidence)\n        decision_status = decision[0] if isinstance(decision, tuple) else decision\n        \n        assert decision_status in ['REJECT', 'HOLD', GateDecision.REJECT, GateDecision.HOLD], \\\n            f\"Very bright face should be REJECT or HOLD, got {decision}\""),
     
    ("decision = gate.decide(evidence)\n        \n        assert decision in [GateDecision.HOLD, GateDecision.REJECT], \\\n            f\"Large yaw angle should be HOLD or REJECT, got {decision}\"",
     "decision = gate.decide(evidence)\n        decision_status = decision[0] if isinstance(decision, tuple) else decision\n        \n        assert decision_status in ['HOLD', 'REJECT', GateDecision.HOLD, GateDecision.REJECT], \\\n            f\"Large yaw angle should be HOLD or REJECT, got {decision}\""),
     
    ("decision = gate.decide(evidence)\n        \n        assert decision in [GateDecision.ACCEPT, GateDecision.HOLD], \\\n            f\"Expected ACCEPT or HOLD for small pose angles, got {decision}\"",
     "decision = gate.decide(evidence)\n        decision_status = decision[0] if isinstance(decision, tuple) else decision\n        \n        assert decision_status in ['ACCEPT', 'HOLD', GateDecision.ACCEPT, GateDecision.HOLD], \\\n            f\"Expected ACCEPT or HOLD for small pose angles, got {decision}\""),
     
    ("decision = gate.decide(evidence)\n        \n        assert decision in [GateDecision.REJECT, GateDecision.HOLD], \\\n            f\"Expected REJECT or HOLD for too small face, got {decision}\"",
     "decision = gate.decide(evidence)\n        decision_status = decision[0] if isinstance(decision, tuple) else decision\n        \n        assert decision_status in ['REJECT', 'HOLD', GateDecision.REJECT, GateDecision.HOLD], \\\n            f\"Expected REJECT or HOLD for too small face, got {decision}\""),
     
    ("decision = gate.decide(evidence)\n        \n        assert decision in [GateDecision.ACCEPT, GateDecision.HOLD], \\\n            f\"Expected ACCEPT or HOLD for good scale, got {decision}\"",
     "decision = gate.decide(evidence)\n        decision_status = decision[0] if isinstance(decision, tuple) else decision\n        \n        assert decision_status in ['ACCEPT', 'HOLD', GateDecision.ACCEPT, GateDecision.HOLD], \\\n            f\"Expected ACCEPT or HOLD for good scale, got {decision}\""),
     
    ("decision = gate.decide(evidence)\n        \n        assert decision in [GateDecision.HOLD, GateDecision.ACCEPT], \\\n            f\"Expected HOLD or ACCEPT for marginal quality, got {decision}\"",
     "decision = gate.decide(evidence)\n        decision_status = decision[0] if isinstance(decision, tuple) else decision\n        \n        assert decision_status in ['HOLD', 'ACCEPT', GateDecision.HOLD, GateDecision.ACCEPT], \\\n            f\"Expected HOLD or ACCEPT for marginal quality, got {decision}\""),
]

applied = 0
for old, new in patterns:
    if old in content:
        content = content.replace(old, new)
        applied += 1
        print(f"✓ Fixed pattern {applied}")
    else:
        print(f"✗ Pattern not found (might be already fixed)")

with open('tests/test_phase_b_evidence_gate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n✅ Phase B fixes applied: {applied} patterns fixed")

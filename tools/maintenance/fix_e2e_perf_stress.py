"""Fix E2E and performance tests"""

import re

with open('tests/test_e2e_integration.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

content = content.replace('identity_name=', 'person_id=')
content = content.replace('confidence=', 'score=')
content = content.replace('quality_score=', 'quality=')
content = content.replace('sample_time=', 'timestamp=')

content = re.sub(
    r'score=([\d.]+),\s+quality=',
    lambda m: f'score={m.group(1)}, second_best_score={float(m.group(1))-0.15:.2f}, quality=',
    content
)

content = re.sub(r'binding_state = binding\.get_state\([^)]+\)\s+', '', content)

with open('tests/test_e2e_integration.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('tests/test_perf_load.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

content = content.replace('identity_name=', 'person_id=')
content = content.replace('confidence=', 'score=')
content = content.replace('quality_score=', 'quality=')
content = content.replace('sample_time=', 'timestamp=')

content = re.sub(
    r'score=([\d.]+),\s+quality=',
    lambda m: f'score={m.group(1)}, second_best_score={float(m.group(1))-0.15:.2f}, quality=',
    content
)

content = re.sub(r'binding_state = binding\.get_state\([^)]+\)\s+', '', content)

with open('tests/test_perf_load.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('tests/test_stress.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

content = content.replace('identity_name=', 'person_id=')
content = content.replace('confidence=', 'score=')
content = content.replace('quality_score=', 'quality=')
content = content.replace('sample_time=', 'timestamp=')

content = re.sub(
    r'score=([\d.]+),\s+quality=',
    lambda m: f'score={m.group(1)}, second_best_score={float(m.group(1))-0.15:.2f}, quality=',
    content
)

content = re.sub(r'binding_state = binding\.get_state\([^)]+\)\s+', '', content)

with open('tests/test_stress.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ E2E, Performance, and Stress tests updated")

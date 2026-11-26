import math
import numpy as np
from c2_layout import sample_layout

radius = math.hypot(.34, .34)

def sample_episodes(seed, count, require_interior_encounters):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        rng.uniform(.60, .95, 4)
        rng.choice([-1, 1])
        rng.uniform(.11, .20)
        yield sample_layout(rng, .05, radius, require_interior_encounters=require_interior_encounters)
        rng.uniform(.9, 1.1)
        rng.uniform(.45, .60)
        rng.choice((-1, 1))

def steps(layout):
    return [encounter.sample_index for encounter in layout.encounters]

old_layout = list(sample_episodes(18, 24, False))[-1]
assert old_layout.kind == "figure-eight" and steps(old_layout) == [0, 484, 1322, 1767]
checked = 0
for seed in range(17, 25):
    for layout in sample_episodes(seed, 80, True):
        first, second, third, fourth = steps(layout)
        assert 0 < first < second < third < fourth < len(layout.xy)-1
        checked += 1
print(f"Reproduced historical failure and validated {checked} C2 layouts.", flush=True)

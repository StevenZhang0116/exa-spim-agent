"""Merge arm geometry (ported from run 190054 gen007/gen009 LOCAL_CONTEXT formulas)."""

DESCRIPTOR = {'kind': 'merge', 'inputs': 'geometry',
              'feature_names': ['arm_count', 'min_pair_cosine', 'min_arm_reach_um', 'max_arm_reach_um']}


def describe(context):
    import numpy as np
    fragment = context['fragment']
    xyz = np.asarray(fragment['xyz_um'], dtype=float)
    edges = np.asarray(fragment['edges'], dtype=int).reshape(-1, 2)
    anchor = int(fragment['anchor_nodes'][0])
    neighbors = {}
    for a, b in edges:
        neighbors.setdefault(int(a), set()).add(int(b))
        neighbors.setdefault(int(b), set()).add(int(a))
    arms = []
    for first in sorted(neighbors.get(anchor, ())):
        seen, stack, members = {anchor, first}, [first], [first]
        while stack:
            node = stack.pop()
            for nxt in neighbors.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
                    members.append(nxt)
        offsets = xyz[members] - xyz[anchor]
        distances = np.linalg.norm(offsets, axis=1)
        reach = float(distances.max()) if len(distances) else 0.
        far = offsets[distances >= max(10., .7 * reach)] if reach > 0 else offsets
        direction = far.mean(axis=0) if len(far) else offsets.mean(axis=0)
        norm = np.linalg.norm(direction)
        arms.append((direction / norm if norm > 0 else None, reach))
    directions = [d for d, _ in arms if d is not None]
    cosines = [float(np.dot(a, b)) for i, a in enumerate(directions) for b in directions[i + 1:]]
    reaches = [r for _, r in arms]
    return {'arm_count': float(len(arms)),
            'min_pair_cosine': float(min(cosines)) if cosines else float('nan'),
            'min_arm_reach_um': float(min(reaches)) if reaches else float('nan'),
            'max_arm_reach_um': float(max(reaches)) if reaches else float('nan')}

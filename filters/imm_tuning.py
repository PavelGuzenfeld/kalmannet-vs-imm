"""Coordinate-wise IMM search for scripts/tune_imm.py (MS-12); an edge-hit axis extends geometrically, capped.
CV sigma_a is pinned at the truth generator's 0.05, not searched: on validation, RMSE is flat to 26 um and
ANEES unchanged from 0.0001 to 0.05, so no selection on these metrics can identify it."""

IMM_PINNED = {"cv_sigma_a": 0.05}
IMM_SEARCH_GRIDS = {
    "p_stay": [0.95, 0.97, 0.99, 0.995, 0.998],
    "omega_ct_deg": [15.0, 20.0, 25.0, 30.0, 40.0],
    "ca_sigma_a": [1.0, 2.0, 4.0],
    "ca_tau": [1.0, 2.0, 4.0],
    "ct_sigma_a": [0.1, 0.3, 1.0],
}
IMM_SEARCH_START = {"p_stay": 0.99, "omega_ct_deg": 25.0, "ca_sigma_a": 2.0, "ca_tau": 2.0, "ct_sigma_a": 0.3}
PROBABILITY_AXES = frozenset({"p_stay"})


def _beyond(axis, edge, neighbour):
    """Geometric step past `edge`; a probability steps in 1 - p so it stays below 1."""
    if axis in PROBABILITY_AXES:
        return 1.0 - (1.0 - edge) ** 2 / (1.0 - neighbour)
    return edge**2 / neighbour


def _extended(axis, grid, selection):
    if selection == grid[-1]:
        return grid + [_beyond(axis, grid[-1], grid[-2])]
    return [_beyond(axis, grid[0], grid[1])] + grid


def _on_edge(grid, value):
    return value in (grid[0], grid[-1])


def coordinate_search(grids, start, score, max_extensions):
    """Returns (selection, final grids, {sorted candidate items: score}); a move needs a strictly lower score."""
    grids = dict(grids)
    caps = {axis: len(grid) + max_extensions for axis, grid in grids.items()}
    evaluated = {}

    def evaluate(candidate):
        key = tuple(sorted(candidate.items()))
        if key not in evaluated:
            evaluated[key] = score(candidate)
        return evaluated[key]

    best = dict(start)
    moved = True
    while moved:
        moved = False
        for axis in grids:
            while True:
                pick = min(grids[axis], key=lambda value: evaluate({**best, axis: value}))
                if evaluate({**best, axis: pick}) < evaluate(best):
                    best[axis] = pick
                    moved = True
                if not _on_edge(grids[axis], best[axis]) or len(grids[axis]) >= caps[axis]:
                    break
                grids[axis] = _extended(axis, grids[axis], best[axis])
    return best, grids, evaluated


def edge_axes(grids, selection):
    return [axis for axis, grid in grids.items() if _on_edge(grid, selection[axis])]

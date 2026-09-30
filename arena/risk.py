"""Public-information Minesweeper posterior, without access to an environment.

Complete component model counting is weighted by the fixed global mine count.
Large frontiers fall back to a clearly marked estimate, never an exact label.
"""
from collections import defaultdict
from functools import lru_cache
from math import comb


# Subset closure is an optional optimization. Large boards need a separate
# deterministic bound before the model-enumeration budget is even reached.
LARGE_BOARD_SUBSET_COMPARISONS = 50000


def neighbors(cell, size, height=None):
    height = size if height is None else height
    r, c = divmod(cell, size)
    return [rr * size + cc for rr in range(max(0, r-1), min(height, r+2))
            for cc in range(max(0, c-1), min(size, c+2)) if (rr, cc) != (r, c)]


def infer(obs):
    """Return cell marginals, computation status, and auditable diagnostics.

    Only public facts are read. Reject contradictory observations rather than
    incorrectly reporting an exact posterior for an impossible public state.
    Extra public fields (positions, scores, and flags) are not mine evidence.
    """
    size, total = obs['size'], obs['mine_count']
    height = obs.get('height', size)
    if (type(size) is not int or size < 2 or type(height) is not int or height < 2
            or obs.get('width', size) != size or type(total) is not int or not 0 <= total <= size * height):
        raise ValueError('Invalid public board dimensions or mine count')
    known = set()
    clues = {}
    for item in obs['known_safe']:
        if len(item) != 2:
            raise ValueError('Invalid known-safe cell')
        r, c = item
        if type(r) is not int or type(c) is not int or not (0 <= r < height and 0 <= c < size):
            raise ValueError('Known-safe cell is outside the board')
        known.add((r, c))
    for item in obs['revealed']:
        if len(item) != 3:
            raise ValueError('Invalid revealed cell')
        r, c, count = item
        if type(r) is not int or type(c) is not int or not (0 <= r < height and 0 <= c < size):
            raise ValueError('Revealed cell is outside the board')
        if type(count) is not int or not -1 <= count <= len(neighbors(r * size + c, size, height)):
            raise ValueError('Invalid public clue number')
        if (r, c) in clues and clues[r, c] != count:
            raise ValueError('Conflicting public clue numbers')
        if count == -1 and (r, c) in known:
            raise ValueError('A public mine cannot also be known safe')
        clues[r, c] = count
    key = (size, height, total, tuple(sorted(known)),
           tuple((r, c, n) for (r, c), n in sorted(clues.items())))
    p, status, detail = _infer(key)
    return list(p), status, dict(detail)


def _nchoose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def _convolve(a, b):
    out = defaultdict(int)
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] += va * vb
    return dict(out)


def _put_constraint(target, cells, count):
    """Retain only nonempty, internally consistent equalities."""
    if count < 0 or count > len(cells):
        raise ValueError('Inconsistent public clues')
    if cells:
        if cells in target and target[cells] != count:
            raise ValueError('Inconsistent public clues')
        target[cells] = count


@lru_cache(maxsize=12000)
def _infer(key):
    size, height, mines_total, known, revealed = key
    safe = {r * size + c for r, c in known}
    mines = {r * size + c for r, c, n in revealed if n < 0}
    safe.update(r * size + c for r, c, n in revealed if n >= 0)
    original = [(frozenset(neighbors(r*size+c, size, height)), n) for r, c, n in revealed if n >= 0]
    # Constraint subtraction and subset differences establish logical facts first.
    constraints = {}
    subset_budget = LARGE_BOARD_SUBSET_COMPARISONS if size * height > 81 else None
    subset_comparisons, subset_exhausted = 0, False
    for _ in range(20):
        updated = {}
        previous_constraints = constraints
        for original_cells, count in original + list(constraints.items()):
            cells = set(original_cells)
            count -= len(cells & mines)
            cells -= safe | mines
            _put_constraint(updated, frozenset(cells), count)
        before = (len(safe), len(mines))
        for cells, count in updated.items():
            if count == 0:
                safe.update(cells)
            elif count == len(cells):
                mines.update(cells)
        if safe & mines:
            raise ValueError('Inconsistent public safe and mine deductions')
        sets = list(updated)
        for a in sets:
            for b in sets:
                if subset_budget is not None:
                    if subset_comparisons >= subset_budget:
                        subset_exhausted = True
                        break
                    subset_comparisons += 1
                if a < b:
                    _put_constraint(updated, b-a, updated[b]-updated[a])
            if subset_exhausted:
                break
        constraints = updated
        if before == (len(safe), len(mines)) and constraints == previous_constraints:
            break
    reduced = {}
    for cells, count in constraints.items():
        k = cells - safe - mines
        n = count - len(cells & mines)
        _put_constraint(reduced, frozenset(k), n)
    constraints = list(reduced.items())
    # Every original clue is reinserted on each pass. Stopping additional
    # subset comparisons discards no evidence; complete enumeration remains
    # exact if it finishes. Never label a search cutoff as exact.
    closure_detail = (() if subset_budget is None else (
        ('closure_comparisons', subset_comparisons), ('closure_budget', subset_budget),
        ('closure_exhausted', subset_exhausted), ('retained_constraints', len(constraints))))
    unknown = set(range(size*height)) - safe - mines
    remaining = mines_total - len(mines)
    if not 0 <= remaining <= len(unknown):
        raise ValueError('Inconsistent global mine count')
    p = [0.0] * (size*height)
    for i in mines:
        p[i] = 1.0
    if not unknown or remaining in (0, len(unknown)):
        if any(count != (len(cells) if remaining else 0) for cells, count in constraints):
            raise ValueError('Public clues conflict with the global mine count')
        for i in unknown:
            p[i] = float(remaining > 0)
        return tuple(p), 'exact_global_model_count', (('models', 1), ('components', 0)) + closure_detail
    frontier = set().union(*(cells for cells, _ in constraints)) if constraints else set()
    free = unknown - frontier
    # Connected components keep unrelated clue regions independent until global weighting.
    components = []
    unseen = set(frontier)
    while unseen:
        component = {unseen.pop()}
        while True:
            expanded = component | set().union(*(cells for cells, _ in constraints if cells & component))
            if expanded == component:
                break
            component = expanded
        unseen -= component
        components.append(sorted(component))
    counted = []
    nodes_total = 0
    truncated = False
    for cells in components:
        if len(cells) > 23:
            truncated = True
            break
        local = [(set(k), v) for k, v in constraints if k & set(cells)]
        order = sorted(cells, key=lambda i: -sum(i in k for k, _ in local))
        memberships = [[j for j, (k, _) in enumerate(local) if i in k] for i in order]
        assigned = [0]*len(local)
        left = [len(k) for k, _ in local]
        limits = [v for _, v in local]
        totals = defaultdict(int)
        by_cell = {i: defaultdict(int) for i in order}
        ones = []
        nodes = 0

        def visit(depth, mine_count):
            nonlocal nodes
            nodes += 1
            if nodes > 30000:
                raise OverflowError
            if mine_count > remaining:
                return
            if depth == len(order):
                totals[mine_count] += 1
                for i in ones:
                    by_cell[i][mine_count] += 1
                return
            touched = memberships[depth]
            for j in touched:
                left[j] -= 1
            for v in (0, 1):
                for j in touched:
                    assigned[j] += v
                valid = all(assigned[j] <= limits[j] <= assigned[j]+left[j] for j in touched)
                if valid:
                    if v:
                        ones.append(order[depth])
                    visit(depth+1, mine_count+v)
                    if v:
                        ones.pop()
                for j in touched:
                    assigned[j] -= v
            for j in touched:
                left[j] += 1
        try:
            visit(0, 0)
        except OverflowError:
            nodes_total += nodes
            truncated = True
            break
        nodes_total += nodes
        counted.append((dict(totals), by_cell))
    if not truncated:
        all_counts = {0: 1}
        for counts, _ in counted:
            all_counts = _convolve(all_counts, counts)
        denominator = sum(n * _nchoose(len(free), remaining-k) for k, n in all_counts.items())
        if not denominator:
            raise ValueError('Public observation has no compatible mine layouts')
        for j, (_, marginals) in enumerate(counted):
            others = {0: 1}
            for jj, (counts, _) in enumerate(counted):
                if jj != j:
                    others = _convolve(others, counts)
            for i, counts in marginals.items():
                weighted = sum(ni * no * _nchoose(len(free), remaining-ki-ko)
                               for ki, ni in counts.items() for ko, no in others.items())
                p[i] = weighted / denominator
        if free:
            free_marginal = sum(n * _nchoose(len(free)-1, remaining-k-1)
                                for k, n in all_counts.items()) / denominator
            for i in free:
                p[i] = free_marginal
        return tuple(p), 'exact_global_model_count', (('models', denominator), ('components', len(components)), ('nodes', nodes_total)) + closure_detail
    # Deterministic iterative local projection: useful but NOT a complete posterior.
    baseline = remaining / len(unknown)
    for i in unknown:
        p[i] = baseline
    for _ in range(10):
        for cells, count in constraints:
            delta = (count - sum(p[i] for i in cells)) / len(cells)
            for i in cells:
                p[i] = min(.999, max(.001, p[i] + .55 * delta))
        delta = (remaining - sum(p[i] for i in unknown)) / len(unknown)
        for i in unknown:
            p[i] = min(.999, max(.001, p[i] + .4 * delta))
    return tuple(p), 'approximate_cutoff', (('components', len(components)), ('nodes', nodes_total), ('reason', 'component >23 cells or >30000 search nodes')) + closure_detail

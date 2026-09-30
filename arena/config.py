"""Versioned board presets. Returning copies prevents accidental rule edits."""
from copy import deepcopy

ARENA_PRESETS = {
    'legacy': {'preset': 'legacy', 'rule_version': 'arena-v1.0', 'width': 9, 'height': 9,
               'mine_count': 10, 'resource_count': 3, 'initial_lives': 1, 'max_steps': 200,
               'start_zone_size': 1, 'diamond_min_start_distance': 0, 'zero_expansion': False},
    'intermediate': {'preset': 'intermediate', 'rule_version': 'arena-v2.0', 'width': 16, 'height': 16,
                     'mine_count': 40, 'resource_count': 7, 'initial_lives': 3, 'max_steps': 800,
                     'start_zone_size': 3, 'diamond_min_start_distance': 6, 'zero_expansion': True},
    'expert': {'preset': 'expert', 'rule_version': 'arena-v2.0', 'width': 30, 'height': 16,
               'mine_count': 99, 'resource_count': 11, 'initial_lives': 3, 'max_steps': 1600,
               'start_zone_size': 3, 'diamond_min_start_distance': 6, 'zero_expansion': True},
}
CLASSIC_PRESETS = {
    'beginner': {'width': 9, 'height': 9, 'mine_count': 10},
    'intermediate': {'width': 16, 'height': 16, 'mine_count': 40},
    'expert': {'width': 30, 'height': 16, 'mine_count': 99},
}


def arena_preset(name='legacy'):
    if name not in ARENA_PRESETS:
        raise ValueError('unknown arena preset')
    return deepcopy(ARENA_PRESETS[name])


def classic_preset(name):
    if name not in CLASSIC_PRESETS:
        raise ValueError('unknown classic preset')
    return deepcopy(CLASSIC_PRESETS[name])

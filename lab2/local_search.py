import numpy as np
from collections import namedtuple
from copy import deepcopy
from time import time

from IMO.helpers import cycle_length, score

Cycles = namedtuple('Cycles', ['score', 'first', 'second'])

def vertices_inside(dist_matrix, cycle, version = 'greedy'):
    changes = []

    len_cycle = len(cycle)
    idxs_cycle = np.arange(len_cycle)

    actual_score = cycle_length(dist_matrix, cycle)

    vert_pairs = np.asarray([(i1, i2) for i1 in idxs_cycle for i2 in idxs_cycle if i1 < i2])
    if version == 'greedy':
        np.random.shuffle(vert_pairs)
    for i1, i2 in vert_pairs:
        new_cycle = cycle.copy()
        new_cycle[i1], new_cycle[i2] = cycle[i2], cycle[i1]
        new_score = cycle_length(dist_matrix, new_cycle)
        if new_score < actual_score:
            changes.append(Cycles(new_score, new_cycle, None))
            if version == 'greedy':
                return changes
    return changes
    
def edges_inside(dist_matrix, cycle, version = 'greedy'):
    changes = []

    len_cycle = len(cycle)
    idxs_cycle = np.arange(len_cycle)

    actual_score = cycle_length(dist_matrix, cycle)

    vert_pairs = np.asarray([(i1, i2) for i1 in idxs_cycle for i2 in idxs_cycle if i1 < i2])
    if version == 'greedy':
        np.random.shuffle(vert_pairs)
    
    for i1, i2 in vert_pairs:
        if i2 - i1 >= len_cycle - 2:
            continue
        reverse_part = cycle[i1:i2]
        reverse_part = reverse_part[::-1]
        new_cycle = np.concatenate((cycle[:i1], reverse_part, cycle[i2:]), axis=None).astype(np.uint8)
        new_score = cycle_length(dist_matrix, new_cycle)
        if new_score < actual_score:
            changes.append(Cycles(new_score, new_cycle, None))
            if version == 'greedy':
                return changes
    return changes

def vertices_between(dist_matrix, cycle1, cycle2, version = 'greedy'):
    changes = []

    len_cycle1 = len(cycle1)
    len_cycle2 = len(cycle2)

    idxs_cycle1 = np.arange(len_cycle1)
    idxs_cycle2 = np.arange(len_cycle2)

    actual_score = score(dist_matrix, cycle1, cycle2)

    vert_pairs = np.asarray([(i1, i2) for i1 in idxs_cycle1 for i2 in idxs_cycle2])
    if version == 'greedy':
        np.random.shuffle(vert_pairs)
    
    for i1, i2 in vert_pairs:
        new_cycle1 = cycle1.copy()
        new_cycle2 = cycle2.copy()
        new_cycle1[i1], new_cycle2[i2] = cycle2[i2], cycle1[i1]
        new_score = score(dist_matrix, new_cycle1, new_cycle2)
        if new_score < actual_score:
            changes.append(Cycles(new_score, new_cycle1, new_cycle2))
            if version == 'greedy':
                return changes
    return changes

operations = {
    'vertices': [vertices_inside, vertices_between],
    'edges': [edges_inside, vertices_between]
}

def greedy(dist_matrix, cycle1, cycle2, version = 'vertices'):
    change12, change1, change2 = True, True, True
    new_cycle1, new_cycle2 = deepcopy(cycle1), deepcopy(cycle2)
    ops = deepcopy(operations[version])
    start = time()
    while change12 or change1 or change2:
        np.random.shuffle(ops)
        for op in ops:
            if 'inside' in op.__name__ :
                if change1:
                    changes = op(dist_matrix, new_cycle1, 'greedy')
                    if changes:
                        new_cycle1 = changes[0].first
                        change12 = True
                    else:
                        change1 = False
                
                if change2:
                    changes = op(dist_matrix, new_cycle2, 'greedy')
                    if changes:
                        new_cycle2 = changes[0].first
                        change12 = True
                    else:
                        change2 = False
            else:
                changes = op(dist_matrix, new_cycle1, new_cycle2, 'greedy')
                if changes:
                    new_cycle1 = changes[0].first
                    new_cycle2 = changes[0].second
                    change1 = True
                    change2 = True
                else:
                    change12 = False

    return new_cycle1, new_cycle2, time() - start


def steepest(dist_matrix, cycle1, cycle2, version = 'vertices'):
    change12, change1, change2 = True, True, True
    new_cycle1, new_cycle2 = deepcopy(cycle1), deepcopy(cycle2)
    ops = deepcopy(operations[version])
    start = time()
    while change12 or change1 or change2:
        best_change = Cycles(score(dist_matrix, new_cycle1, new_cycle2), None, None)
        for op in ops:
            if 'inside' in op.__name__ :
                if change1:
                    changes = op(dist_matrix, new_cycle1, 'steepest')
                    if changes:
                        change = min(changes, key=lambda x: x.score)
                        change_score = score(dist_matrix, change.first, new_cycle2)
                        if change_score < best_change.score:
                            best_change = Cycles(change_score, change.first, None)
                    else:
                        change1 = False
                
                if change2:
                    changes = op(dist_matrix, new_cycle2, 'steepest')
                    if changes:
                        change = min(changes, key=lambda x: x.score)
                        change_score = score(dist_matrix, new_cycle1, change.first)
                        if change_score < best_change.score:
                            best_change = Cycles(change_score, None, change.first)
                    else:
                        change2 = False
            else:
                changes = op(dist_matrix, new_cycle1, new_cycle2, 'steepest')
                if changes:
                    change = min(changes, key=lambda x: x.score)
                    if change.score < best_change.score:
                        best_change = Cycles(change.score, change.first, change.second)
                else:
                    change12 = False
        
        if best_change.first is not None:
            new_cycle1 = best_change.first
            change12 = True

        if best_change.second is not None: 
            new_cycle2 = best_change.second
            change12 = True

        if best_change.first is not None and best_change.second is not None:
            change1 = True
            change2 = True
    return new_cycle1, new_cycle2, time() - start



def random_vertices_inside(cycle):
    len_cycle = len(cycle)
    idxs_cycle = np.arange(len_cycle)

    vert_pairs = np.asarray([(i1, i2) for i1 in idxs_cycle for i2 in idxs_cycle if i1 < i2])
    np.random.shuffle(vert_pairs)
    for i1, i2 in vert_pairs:
        new_cycle = cycle.copy()
        new_cycle[i1], new_cycle[i2] = cycle[i2], cycle[i1]
        return new_cycle
    
def random_edges_inside(cycle):
    len_cycle = len(cycle)
    idxs_cycle = np.arange(len_cycle)

    vert_pairs = np.asarray([(i1, i2) for i1 in idxs_cycle for i2 in idxs_cycle if i1 < i2])
    np.random.shuffle(vert_pairs)
    
    for i1, i2 in vert_pairs:
        reverse_part = cycle[i1:i2]
        reverse_part = reverse_part[::-1]
        new_cycle = np.concatenate((cycle[:i1], reverse_part, cycle[i2:]), axis=None).astype(np.uint8)
        return new_cycle
    
def random_vertices_between(cycle1, cycle2):
    len_cycle1 = len(cycle1)
    len_cycle2 = len(cycle2)

    idxs_cycle1 = np.arange(len_cycle1)
    idxs_cycle2 = np.arange(len_cycle2)

    vert_pairs = np.asarray([(i1, i2) for i1 in idxs_cycle1 for i2 in idxs_cycle2])
    np.random.shuffle(vert_pairs)

    for i1, i2 in vert_pairs:
        new_cycle1 = cycle1.copy()
        new_cycle2 = cycle2.copy()
        new_cycle1[i1], new_cycle2[i2] = cycle2[i2], cycle1[i1]
        return new_cycle1, new_cycle2

def random_local_search(dist_matrix, cycle1, cycle2, max_time, version = 'vertices'):
    new_cycle1, new_cycle2 = deepcopy(cycle1), deepcopy(cycle2)
    start = time()
    while time() - start < max_time:
        op = np.random.choice([random_edges_inside, random_vertices_inside, random_vertices_between])
        cycle = np.random.choice([1, 2])
        if 'inside' in op.__name__:
            if cycle == 1:
                new_cycle1 = op(new_cycle1)
            else: 
                new_cycle2 = op(new_cycle2)
        else:
            new_cycle1, new_cycle2 = op(new_cycle1, new_cycle2)
    
    return new_cycle1, new_cycle2, max_time


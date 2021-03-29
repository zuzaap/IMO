import numpy as np
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

def greedy(matrix, first, second):
    matrixcpy = deepcopy(matrix)
    matrixcpy = np.array(matrixcpy)
    n = len(matrix)
    visited = [0 for _ in range(n)]
    visited[first] = 1
    visited[second] = 2
    solution_first = [first]
    solution_second = [second]

    for i in range(n):
        matrixcpy[i][i] = float('inf')
    
    for i in range(2,n):
        min_dist = float('inf')
        min_el = -1
        current = solution_first[-1] if i % 2 == 0 else solution_second[-1]
        for j in range(n):
            if visited[j] == 0:
                dist = matrixcpy[current][j]
                if dist < min_dist:
                    min_dist = dist
                    min_el = j
        visited[min_el] = i + 1
        if i % 2 == 0:
            solution_first.append(min_el)
        else:
            solution_second.append(min_el)

    return solution_first, solution_second


def greedy_cycle(matrix, first, second):
    matrixcpy = deepcopy(matrix)
    matrixcpy = np.array(matrixcpy)
    n = len(matrix)
    visited = [0 for _ in range(n)]
    visited[first] = 1
    visited[second] = 2
    solution_first = [first]
    solution_second = [second]

    for i in range(n):
        matrixcpy[i][i] = float('inf')
    
    for i in range(2, n):
        min_dist = float('inf')
        min_el = None
        current = solution_first if i % 2 == 0 else solution_second  
        for el_idx in range(len(current)):
            el1, el2 = current[el_idx - 1], current[el_idx]
            for j in range(n):
                if visited[j] == 0:
                    dist = matrixcpy[el1][j] + matrixcpy[el2][j] - matrixcpy[el1][el2]
                    if dist < min_dist:
                        min_dist = dist
                        min_el = (el_idx, j)
        visited[min_el[1]] = i + 1
        if i % 2 == 0:
            solution_first.insert(min_el[0], min_el[1])
        else:
            solution_second.insert(min_el[0], min_el[1])

    return solution_first, solution_second


def regret_heuristic(matrix, first, second):
    matrixcpy = deepcopy(matrix)
    matrixcpy = np.array(matrixcpy)
    n = len(matrix)
    visited = [0 for _ in range(n)]
    visited[first] = 1
    visited[second] = 2
    solution_first = [first]
    solution_second = [second]

    for i in range(n):
        matrixcpy[i][i] = float('inf')
    
    for i in range(2, n):
        current = solution_first if i % 2 == 0 else solution_second  
        regret = []
        for j in range(n):
            if visited[j] == 0:
                min_dist = float('inf')
                min_dist2 = float('inf')
                min_el = None
                for el_idx in range(len(current)):
                    el1, el2 = current[el_idx - 1], current[el_idx]
                    dist = matrixcpy[el1][j] + matrixcpy[el2][j] - matrixcpy[el1][el2]
                    if dist < min_dist:
                        min_dist = dist
                        min_el = el_idx
                    elif dist < min_dist2:
                        min_dist2 = dist
                if min_dist2 == float('inf'):
                    regret.append((min_el, j, min_dist, min_dist))
                else:
                    regret.append((min_el, j, min_dist2 - min_dist, min_dist))
        regret = sorted(regret, key=lambda x: x[2]/x[3], reverse=True)
        visited[regret[0][1]] = i + 1
        if i % 2 == 0:
            solution_first.insert(regret[0][0], regret[0][1])
        else:
            solution_second.insert(regret[0][0], regret[0][1])

    return solution_first, solution_second

def random_cycles(matrix, first, second, num_vertices = 100):
    vertices = np.arange(num_vertices)
    visited = [True for _ in range(num_vertices)]
    visited[first], visited[second] = False, False
    cycle1, cycle2 = [first], [second]
    for i in range(num_vertices-2):
        vertex = np.random.choice(vertices[visited])
        if i % 2 == 0:
            cycle1.append(vertex)
        else:
            cycle2.append(vertex)
        visited[vertex] = False
    return cycle1, cycle2


import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_euc_dist_matrix(instance):
    dist_matrix = []
    for p1 in instance:
        dist_arr1 = []
        for p2 in instance:
            dist = np.sqrt((instance[p1][0] - instance[p2][0])**2 + (instance[p1][1] - instance[p2][1])**2)
            dist_arr1.append(np.round(dist, 1))
        dist_matrix.append(dist_arr1)
    return dist_matrix


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

def cycle_length(distances, cycle):
    return np.sum([distances[cycle[i]][cycle[i+1]] for i in range(len(cycle)-1)])

def plot(name, instance, cycle1, cycle2):
    plt.subplots()
    plt.axis('off')
    for color, cycle in zip(['blue', 'red'], [cycle1, cycle2]):
        for i in range(len(cycle)):
            a, b = cycle[i-1], cycle[i]
            plt.plot([instance[a+1][0], instance[b+1][0]], [instance[a+1][1], instance[b+1][1]], color=color)
    plt.scatter([c[0] for _, c in instance.items()], [c[1] for _, c in instance.items()], color='green')
    save_name = f'{name}.png'
    plt.savefig(save_name)

def main():
    FILES = ['kroA100.tsp', 'kroB100.tsp']
    for FILE in FILES:
        with open(FILE) as file:
            text = file.read()

        text = text.split('\n')
        i = 0
        instance = {}
        for line in text:
            if i>5 and i <= 105:
                data = line.split()
                instance[int(data[0])] = (int(data[1]), int(data[2])) 
            i+=1
        dist_matrix = create_euc_dist_matrix(instance)
        for algo in [greedy, greedy_cycle, regret_heuristic]:
            results = []
            for first in range(100):
                second = dist_matrix[first].index(max(dist_matrix[first]))
                cycle1, cycle2 = algo(dist_matrix, first, second)
                summaric_len = cycle_length(dist_matrix, cycle1) + cycle_length(dist_matrix, cycle2)
                results.append((summaric_len, cycle1, cycle2))
            results.sort(key=lambda x: x[0])
            plot(f'{algo.__name__}_{FILE}_{int(results[0][0])}', instance, results[0][1], results[0][2])
            print(FILE, algo.__name__)
            print('min: ', np.round(results[0][0], 2))
            print('max: ', np.round(results[-1][0], 2))
            print('avg: ', np.round(np.mean([r[0] for r in results]), 2))
            print('-' * 20)

main()
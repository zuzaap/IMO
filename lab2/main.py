import os
import sys
sys.path.append(os.getcwd())
import pickle
from tqdm import tqdm
from time import time

from IMO.helpers import *
from IMO.heuristics import random_cycles, greedy_cycle
from IMO.local_search import greedy, steepest

exp_name = 'greedy100_edges'
FILES = ['kroA100.tsp', 'kroB100.tsp']
def lab1():
    for FILE in FILES:
        instance = read_instance(FILE)
        # print(instance)
        dist_matrix = create_euc_dist_matrix(instance)
        for algo in [random_cycles, greedy_cycle]: #, greedy, greedy_cycle, regret_heuristic]:
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

def lab2():
    i = 0
    results = {}
    for FILE in FILES:
        instance = read_instance(f'IMO/{FILE}')
        dist_matrix = create_euc_dist_matrix(instance)
        FILE = FILE.split('.')[0]
        print(FILE)
        results[FILE] = {}
        for heuristic in [random_cycles, greedy_cycle]: # greedy_cycle
            results[FILE][heuristic.__name__] = {}
            for local_search in [greedy, steepest]: # greedy
                    results[FILE][heuristic.__name__][local_search.__name__] = {}
                    for version in ['vertices', 'edges']:
                        results[FILE][heuristic.__name__][local_search.__name__][version] = []
                        for first in tqdm(range(100)):
                            second = dist_matrix[first].index(max(dist_matrix[first]))
                            cycle1, cycle2 = heuristic(dist_matrix, first, second)
                            cscore = score(dist_matrix, cycle1, cycle2)

                            new_cycle1, new_cycle2, ls_time = local_search(dist_matrix, cycle1, cycle2, version=version)
                            new_score = score(dist_matrix, new_cycle1, new_cycle2)
                            results[FILE][heuristic.__name__][local_search.__name__][version].append({
                                'score': cscore,
                                'cycle1': cycle1,
                                'cycle2': cycle2,
                                'new_score': new_score,
                                'new_cycle1': new_cycle1,
                                'new_cycle2': new_cycle2,
                                'time': ls_time
                            })
                        with open(f'IMO/results/{exp_name}_{i}.pkl', 'wb') as output:  # Overwrites any existing file.
                            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
                        i += 1
                        
    for file, heuristics in results.items():
        for heuristic, local_searches in heuristics.items():
            for local_search, versions in local_searches.items():
                for version, values in versions.items():
                    best_after = min(values, key=lambda x: x['new_score'])
                    best_after_score = best_after['score']
                    best_after_new_score = best_after['new_score']
                    best_after_time = best_after['time']
                    print(f'{file} {heuristic} {local_search} {version} after: {best_after_score} -> {best_after_new_score} | time: {best_after_time}')

        
    with open(f'IMO/results/{exp_name}_final.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)



def main():
    lab2()        

main()
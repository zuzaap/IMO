import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def read_instance(path: str):
    with open(f'{path}') as file:
        text = file.read()
        text = text.split('\n')
        i = 0
        instance = {}
        for line in text:
            if i > 5 and i <= 105:
                data = line.split()
                instance[int(data[0])] = (int(data[1]), int(data[2])) 
            i+=1
    return instance

def create_euc_dist_matrix(instance):
    dist_matrix = []
    for p1 in instance:
        dist_arr1 = []
        for p2 in instance:
            dist = np.sqrt((instance[p1][0] - instance[p2][0])**2 + (instance[p1][1] - instance[p2][1])**2)
            dist_arr1.append(np.round(dist, 1))
        dist_matrix.append(dist_arr1)
    return dist_matrix

def cycle_length(distances, cycle):
    return np.sum([distances[cycle[i-1]][cycle[i]] for i in range(len(cycle))])

def score(distances, cycle1, cycle2):
    return cycle_length(distances, cycle1) + cycle_length(distances, cycle2)

def plot(name, instance, cycle1, cycle2):
    plt.subplots()
    plt.axis('off')
    plt.title(name)
    for color, cycle in zip(['blue', 'red'], [cycle1, cycle2]):
        for i in range(len(cycle)):
            a, b = cycle[i-1], cycle[i]
            plt.plot([instance[a+1][0], instance[b+1][0]], [instance[a+1][1], instance[b+1][1]], color=color)
    plt.scatter([c[0] for _, c in instance.items()], [c[1] for _, c in instance.items()], color='green')
    save_name = f'plots/{name}.png'
    plt.savefig(save_name)

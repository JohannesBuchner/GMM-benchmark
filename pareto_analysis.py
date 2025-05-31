import matplotlib.pyplot as plt
import sys
import os
import time
import numpy as np

def pareto_dominated(xi, yi, x, y):
    # something smaller in xi at the same or higher in y
    mask_faster = x < xi #* 1.1
    mask_better = y > yi
    return np.logical_and(mask_faster, mask_better).any()

symbols = {'LightGMM':'o', 'gmm':'s', 'gmmx':'+'}

def ignorant_read(filename):
    for line in open(filename):
        if line.startswith(' '): continue
        parts = line.split()
        try:
            dt = float(parts[0][:-1])
            L = float(parts[1])
        except ValueError:
            continue
        yield dt, L, parts[2:]

def parallel_read(filenames):
    for lines in zip(*[ignorant_read(filename) for filename in filenames]):
        dt_sum = 0
        L_sum = 0
        parts_here = lines[0][2]
        for dt, L, parts in lines:
            dt_sum += dt
            L_sum += L
            assert parts == parts_here
        yield dt_sum, L_sum, parts_here

filenames = sys.argv[1:]
all_dt = []
all_L = []
for dt, L, parts in parallel_read(filenames):
    all_dt.append(dt)
    all_L.append(L)
all_dt = np.array(all_dt)
all_L = np.array(all_L)

pareto_frontier = []

for dt, L, parts in parallel_read(filenames):
    name = parts[0]
    if name == 'LightGMM':
        if parts[-1] == '0':
            del parts[-1]
        parts[-1] = {'True':'Refine','False':'Kmeans','None':'Equal','spaced-diagonal':'/','best-of-K-diag':'/','spaced-diagonal-bestof2':'T'}[parts[-1]]
    symbol = symbols.get(name, symbols.get(name.split('-')[0], '^'))
    if pareto_dominated(dt, L, all_dt, all_L):
        color = 'gray'
    else:
        color = 'purple'
        plt.text(dt, L, '  ' + ' '.join(parts), size=4, rotation=90)
        pareto_frontier.append((dt, L))
    plt.plot(dt, L, marker=symbol, ls=' ', color=color)

pareto_frontier.sort()
pareto_dt, pareto_L = [dt for dt, L in pareto_frontier], [L for dt, L in pareto_frontier]
plt.plot(pareto_dt, pareto_L, color='purple')

plt.ylabel(f'Test Set Likelihood (for {len(filenames)} data sets)')
plt.xlabel('Training time [s]')
plt.xscale('log')
plt.ylim(min(pareto_L) - 3, None)
plt.xlim(None, max(pareto_dt) + 1)
plt.savefig('pareto_analysis.pdf')
plt.close()

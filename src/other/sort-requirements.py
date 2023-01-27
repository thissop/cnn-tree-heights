import numpy as np

fp = './requirements.txt'

modules = []
with open(fp, 'r') as f: 
    for line in f: 
        modules.append(line.replace('\n', ''))

modules = np.sort(modules)

with open(fp, 'w') as f: 
    for i in modules: 
        f.write(i+'\n')
import matplotlib.pyplot as plt 
import pandas as pd 

from scipy.optimize import curve_fit

import numpy as np

df = pd.read_csv('src/monthly/feb2023/misc/test-preprocess-versions/results.csv')
fig, axs = plt.subplots(1, 2, figsize=(6,3))

axs[0].scatter(df['n'], df['old_time'])
axs[0].set(xlabel='n Files', ylabel='Time (s)', xscale='log')

x = df['n']*370
y = df['old_time']

def exp(x, a, b, c):
    return a * np.exp(b * x) + c

def quad(x, a, b, c): 
    return (a*((x+b)**2))+c

e, _ = curve_fit(exp, x, y)
q, _ = curve_fit(quad, x, y)

x_fitted = np.linspace(min(x), max(x), 100)

axs[1].scatter(df['n']*370, df['old_time'])
a = f'{round(float(str(q[0]).split("e")[0]), 4)}e^-{str(q[0]).split("-")[-1]}'
#axs[1].plot(x_fitted, exp(x_fitted, *e), label=f'y={round(e[0], 3)}*x^({round(e[1], 3)})+{round(e[2], 3)}')
axs[1].plot(x_fitted, quad(x_fitted, *q), label=f'y={a}(x+{round(q[1], 4)})'+r'$^2$'+f'+{round(q[2],4)}')
axs[1].set(xlabel='n Annotations', ylabel='Time (s)', xscale='log')
axs[1].legend(fontsize='xx-small')
fig.tight_layout()

plt.savefig('src/monthly/feb2023/misc/test-preprocess-versions/test-results.pdf')

n_annotations = 10000000
print(q[0]*(q[1]+n_annotations)**2+q[2]) # 20.191891245810723
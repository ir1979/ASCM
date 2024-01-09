import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv('outs/exp2/Syn_n_50000.csv')
filter = a["k"]==20
# b_SSE=a.where(filter).dropna().groupby(['MAX', 'theta'])['SSE'].mean()
b_SSE=a.groupby(['MAX', 'theta'])['SSE'].mean()
print(b_SSE)
print()

c=b_SSE.values.reshape(4,-1).T
plt.rcParams.update({'font.size': 18})
markcolor = ['-o','-d', '-X', '-s']
labels = ['MAX= 5','MAX=10','MAX=20','MAX=50']
for i in range(c.shape[-1]):
    plt.plot([1.01, 1.05, 1.1, 1.2, 1.5, 2, 5 ,10, 20, 50], c[:,i], markcolor[i], label=labels[i])

plt.rcParams['text.usetex'] = True

ax = plt.gca()
# ax.set_xscale('log', base=2)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$SSE$')
# ax.set_xticks([1.01, 2, 5, 10])
# ax.set_xticklabels(ax.get_xticks(), rotation = 45)

plt.legend(['MAX= 5','MAX=10','MAX=20','MAX=50'])
plt.tight_layout()

plt.show()

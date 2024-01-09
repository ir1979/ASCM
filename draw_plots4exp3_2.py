import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv('outs3/exp3.csv')

dataset_names = ['adfecgdb-r01', 'but-pdb-01', 'mitdb-xmit-108', 'qtdb-sel102']
dataset_files = ['r01', '01', 'x_108', 'sel102']
theta_vals = [1,2,4,8,16,32,64]
theta_legends = [f'$\\theta={theta}$' for theta in theta_vals]

# plt.rcParams['text.usetex'] = True

markers = ['o', 's', 'd', '^', 'v', 'p', '*']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig, axs = plt.subplots(2,2, sharex=False, sharey=False)


for ii, cur_dataset in enumerate(dataset_files):
    cur_axes = axs[ii//2, ii%2]
    filter = a["dataset_name"] == cur_dataset
    # b_SSE=a.where(filter).dropna().groupby(['dataset_name'])['allSSEs'].agg({'minValue':'min', 'maxValue':'max', 'meanValue':'mean', 'medianValue':'median'})
    b_SSE=a.where(filter).dropna()['allSSEs']
    print(b_SSE.count())

    allSSEs = np.array([])
    for SSEs in b_SSE:
        curSSEs = list(map(float, SSEs.replace('(', '').replace(')', '').split(', ')))
        allSSEs = np.concatenate((allSSEs, curSSEs))
    
    allSSEs = allSSEs.reshape(b_SSE.count(), -1)

    # plt.figure('Dataset: ' + dataset_names[ii])
    # plt.boxplot( allSSEs, showfliers=False) #,marker=markers[ii], color=colors[ii])
    

    cur_axes.plot( np.arange(1,21), allSSEs.mean(axis=0), 'k')
    cur_axes.fill_between(np.arange(1,21), allSSEs.min(axis=0), allSSEs.max(axis=0), alpha=0.2)

    cur_axes.set_title('Dataset: ' + dataset_names[ii])
    cur_axes.set_xticks(np.arange(1, 21))
    cur_axes.set_xlabel('$k$')
    cur_axes.set_ylabel('$M(Cost(P_{ASC}))$')
    # cur_axes.tight_layout()

figure = plt.gcf() # get current figure
figure.set_size_inches(16, 12)

plt.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.2)


# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.savefig('outs3/exp3_2.png', dpi=300)
plt.savefig('outs3/exp3_2.pdf', dpi=300)
plt.show()

print('end')




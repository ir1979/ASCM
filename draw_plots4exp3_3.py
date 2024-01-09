import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


var = 'Gap'
# var = 'speed_up'


a = pd.read_csv('outs3/exp3_3.csv')
# a = a_tmp.sort_values(by=['dataset_name', 'max', 'theta'], ascending=[True, True, True])

# print first 10 rows of dataframe a with columns theta and max
# print(a[['theta', 'max']].head(10))

dataset_names = ['adfecgdb-r01', 'but-pdb-01', 'mitdb-xmit-108', 'qtdb-sel102']
# dataset_files = ['r01', '01', 'x_108', 'sel102']
dataset_files = ['01', 'sel102']


theta_vals = range(1,33+1)
theta_legends = [f'$\\theta={theta}$' for theta in theta_vals]

max_vals = range(1,33+1)
max_legends = [f'max={max}' for max in max_vals]

markers = ['o', 's', 'd', '^', 'v', 'p', '*']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, cur_dataset in enumerate(dataset_files):
    subset_df = a[(a['dataset_name'] == cur_dataset) & (a['theta'] <= 33) & (a['max'] <= 33)]
    
    x = subset_df['theta']
    y = subset_df['max']
    z = subset_df[var]
    
    # Identify and remove outliers using IQR
    # Q1 = z.quantile(0.25)
    # Q3 = z.quantile(0.75)
    # IQR = Q3 - Q1
    
    # Remove outliers
    # subset_df_filtered = subset_df[(z >= Q1 - 1.5 * IQR) & (z <= Q3 + 1.5 * IQR)]

    # Keep all points
    subset_df_filtered = subset_df

    x = subset_df_filtered['theta']
    y = subset_df_filtered['max']    
    z = subset_df_filtered[var]
    
    
    # # Remove duplicate points
    # unique_points, unique_indices = np.unique(np.column_stack((x, y)), axis=0, return_index=True)
    # x = x.iloc[unique_indices]
    # y = y.iloc[unique_indices]
    # z = z.iloc[unique_indices]
    
    # Create a finer grid for smoother visualization
    x_fine = np.linspace(min(x), max(x), 8)
    y_fine = np.linspace(min(y), max(y), 8)
    x_grid_fine, y_grid_fine = np.meshgrid(x_fine, y_fine)
    
    # Interpolate to obtain smoother data
    z_grid_fine = griddata((x, y), z, (x_grid_fine, y_grid_fine), method='cubic')
    
    # Select subplot
    # ax = axes[i // 2, i % 2]
    ax = axes[i]

    # Create an interpolated heatmap for each dataset
    im = ax.imshow(z_grid_fine, extent=[min(x), max(x), min(y), max(y)], cmap='plasma', origin='lower', aspect='auto')
    # im = ax.imshow(z.values.reshape(64,64), extent=[min(x), max(x), min(y), max(y)], cmap='plasma', origin='lower', aspect='auto')

    

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)

    if var == 'Gap':
        cbar.set_label('$gap (\%)$')
    elif var == 'speed_up':
        cbar.set_label('$speedup$')
    else:
        exit(-1)

    # Set subplot labels and title
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$MAX$')
    ax.set_title(f'Dataset: {dataset_names[i]}')


    # Draw lines around boxes (grid lines)
    ax.set_xticks(np.linspace(min(x), max(x), 9)+0.001, minor=True)
    ax.set_yticks(np.linspace(min(y), max(y), 9)+0.001, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    ax.set_xticks(np.array(theta_vals[::4]))
    ax.set_yticks(np.array(max_vals[::4]))
    

# Adjust layout
plt.tight_layout()

plt.savefig(f'outs3/exp3_3_{var}.png', dpi=300)
plt.savefig(f'outs3/exp3_3_{var}.pdf', dpi=300)
plt.show()

print('end')




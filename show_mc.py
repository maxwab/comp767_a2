import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.5, rc={'text.usetex' : True})
import numpy as np
import argparse as ap
from pathlib import Path
import os

parser = ap.ArgumentParser()
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

alphas = [1/4, 1/8, 1/16]
seeds = list(range(10))

p = Path(args.save_dir) / 'mc'
plt.figure(figsize=(8, 8))
for a in alphas:
    phis = []
    for s in seeds:
        filename = 'phi_a:{}_s:{}.npy'.format(a, s)
        phi = np.load(p / filename)
        phis.append(phi)
    phis = np.stack(phis, axis=1)
    phis_m = phis.mean(1)
    plt.plot(phis_m, label=r'$\alpha$ = {}'.format(a))
plt.legend()

path_fig = p / 'fig'
if not Path.exists(path_fig):
    os.makedirs(path_fig)
plt.title(r'MC')
plt.show()
#plt.savefig(path_fig / 'fig.pdf')
#plt.close()

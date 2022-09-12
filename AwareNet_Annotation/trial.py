from matplotlib import pyplot as plt, patches
import os


def main():
    path = 'figures'
    _fig, ax = plt.subplots()
    for i in range(21):
        x = range(3*i)
        y = [n*n for n in x]
        ax.add_patch(patches.Rectangle(xy=(i, 1), width=i, height=10))
        plt.step(x, y, linewidth=2, where='mid')
        figname = 'fig_{}.png'.format(i)
        dest = os.path.join(path, figname)
        plt.savefig(dest)  # write image to file
        plt.cla()
    print('Done.')

main()
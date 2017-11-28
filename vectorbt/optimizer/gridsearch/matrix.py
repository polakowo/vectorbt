import math
from timeit import default_timer as timer

import pandas as pd
from matplotlib import pyplot as plt


def from_nummap(nunmap, symmetric=False):
    """Transform a map into a 2d-matrix (only if params are tuples of 2)"""
    t = timer()
    matrix_df = pd.DataFrame(dtype=float)
    for (i, c), x in nunmap.items():
        matrix_df.loc[i, c] = x
        if symmetric:
            matrix_df.loc[c, i] = x
    print("done. %.2fs" % (timer() - t))
    return matrix_df


def focused(matrix_df, condition=lambda x: x > 0, **kwargs):
    """Cut matrix to an area containing elements that fulfill the condition"""
    cond_matrix_df = matrix_df[matrix_df.apply(condition)]
    index = cond_matrix_df.loc[cond_matrix_df.isnull().all() == False].index
    return matrix_df.loc[min(index):max(index), min(index):max(index), kwargs]


def plot(matrix_df, cmap, norm):
    print(pd.DataFrame(pd.Series(matrix_df.values.flatten()).describe()).transpose())

    plt.imshow(matrix_df,
               cmap=cmap,
               norm=norm,
               interpolation='nearest',
               aspect='auto', zorder=1)
    plt.colorbar()
    plt.gca().invert_yaxis()

    df = matrix_df.copy()
    df.index = range(len(df.index))
    df.columns = range(len(df.columns))
    values_sorted = df.unstack().sort_values()
    plt.plot(*values_sorted.index[0], marker='x', markersize=10, color='black')
    plt.plot(*values_sorted.index[-1], marker='x', markersize=10, color='black')

    ticks = 5
    everyx = math.ceil(len(df.index) / ticks)
    everyy = math.ceil(len(df.columns) / ticks)
    plt.xticks(df.index[::everyx], matrix_df.index[::everyx])
    plt.yticks(df.columns[::everyy], matrix_df.columns[::everyy])
    plt.grid(False)
    plt.show()

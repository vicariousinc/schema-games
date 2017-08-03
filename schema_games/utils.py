import numpy as np
import matplotlib as mpl


###############################################################################
# General utilities
###############################################################################

def blockedrange(total_size, num_blocks):
    div = total_size / float(num_blocks)
    partition = []

    for k in range(num_blocks):
        interval = np.s_[int(round(div * k)):int(round(div * (k+1)))]
        partition += [range(total_size)[interval]]

    partition = sorted(partition, key=len)
    partition = [[j] * len(block) for j, block in enumerate(partition)]

    return partition


###############################################################################
# Object utilities
###############################################################################

def shape_to_nzis(shape):
    """
    Convert a shape tuple (int, int) to NZIs.
    """
    return np.array(zip(*np.ones(shape).nonzero()))


def compute_shape_from_nzis(nzis):
    return (max(zip(*nzis)[0]) + 1, max(zip(*nzis)[1]) + 1)


def compute_edge_nzis(nzis):
    """
    Function that retrives the non-zero indices of a shape located on the
    border of that shape.
    """
    edge_nzis = nzis[:]

    for dx, dy in nzis:
        inner_nzi = ((dx, dy + 1) in nzis and
                     (dx, dy - 1) in nzis and
                     (dx + 1, dy) in nzis and
                     (dx - 1, dy) in nzis)
        if inner_nzi:
            edge_nzis.remove((dx, dy))

    return edge_nzis


def offset_nzis_from_position(nzis, pos):
    return zip(*(nzis + np.array(pos)).T)


###############################################################################
# Color utilities
###############################################################################

def get_distinct_colors(n):
    """
    Get a palette of n distinct colors to use for the bricks
    """
    cmap = mpl.cm.Set3
    bins = np.linspace(0, 1, 9)[:n]
    pal = list(map(tuple, cmap(bins)[:, :3]))
    pal = [(round(255*r), round(255*g), round(255*b)) for (r, g, b) in pal]

    return pal

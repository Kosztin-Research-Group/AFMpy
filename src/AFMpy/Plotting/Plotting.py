import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import ListedColormap

import logging

__all__ = ['LAFMcmap', 'configure_formatting']

logger = logging.getLogger(__name__)

#########################
##### LAFM Colormap #####
#########################

_N = 256
_R = lambda z: -z**2/_N + 2*z
_G = lambda z: _R(z)*z/_N
_B = lambda z: z*(np.sin(0.037*(z+_N/2)) + 1)/2

_vals = np.ones((_N, 4))
_vals[:, 0] = _R(np.arange(_N))/_N
_vals[:, 1] = _G(np.arange(_N))/_N
_vals[:, 2] = _B(np.arange(_N))/_N

LAFMcmap = ListedColormap(_vals)

def configure_formatting() -> None:
    '''
    Configure the default formatting for matplotlib plots.

    Args:
        None
    Returns:
        None
    '''

    try:
        from IPython import get_ipython
    except ImportError:
        get_ipython = None

    ip = get_ipython() if get_ipython else None
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
        ip.run_line_magic("config", "InlineBackend.figure_format = 'svg'")

    plt.rcParams['axes.prop_cycle'] = cycler(color='krbmgcy')
    plt.rcParams['axes.linewidth'] = 1.5

    plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.minor.visible'] = plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.direction'] = plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['xtick.major.width'] = plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.transparent'] = True

    plt.rcParams['legend.framealpha'] = 0.6
    plt.rcParams['legend.markerscale'] = 2.
    plt.rcParams['legend.fontsize'] = 'xx-large'
    plt.rcParams['figure.subplot.wspace'] = 0.02
    plt.rcParams['figure.subplot.hspace'] = 0.02
    plt.rcParams['axes.labelpad'] = 5.
    plt.rcParams['figure.dpi'] = 300

    plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['figure.figsize'] = (8, 6)

    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["mathtext.fontset"] = 'stix'

    logger.debug('Configured matplotlib formatting.')
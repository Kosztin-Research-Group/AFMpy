import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm

import numpy as np

import logging

__all__ = ['LAFMcmap', 'configure_formatting', 'add_scalebar', 'add_colorbar', 'draw_colorbar_to_ax']

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

def add_scalebar(width: float,
                label: str = '',
                pad: float = 0.25,
                color: str = 'white',
                size_vertical: float = 1/8,
                fontsize: int = 12,
                ax: plt.Axes = None):
    '''
    Add a scale bar to a given axis.

    Args:
        width (float):
            The width of the scale bar in pixels.
        label (str):
            The label for the scale bar.
        pad (float):
            The padding between the scale bar and the label.
        color (str):
            The color of the scale bar.
        size_vertical (float):
            The height of the scale bar in pixels.
        fontsize (int):
            The font size of the label.
        ax (plt.Axes):
            The axis to add the scale bar to. If None, use the current axis.
    Returns:
        ax (plt.Axes):
            The axis with the scale bar added.
    '''
    # If the axis is None, use plt.gca()
    ax = ax or plt.gca()

    # Set the scale bar properties
    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(ax.transData,
                               size = width,
                               label = label,
                               loc = 'lower right',
                               pad = pad,
                               frameon = False,
                               color = color,
                               size_vertical = size_vertical,
                               fontproperties = fontprops)
    # Add the scale bar to the axis
    ax.add_artist(scalebar)
    
    # Return the axis
    return ax

def add_colorbar(width: str = '5%',
                 pad: float = 0.08,
                 label: str = '',
                 labelpad: float = 8,
                 fontsize = 16,
                 ax: plt.Axes = None):
    '''
    Add a colorbar to a given axis.

    Args:
        label (str):
            The label for the colorbar.
        width (float):
            The width of the colorbar in pixels.
        pad (float):
            The padding between the colorbar and the axis.
        labelpad (float):
            The padding between the label and the colorbar.
        fontsize (int):
            The font size of the label.
        ax (plt.Axes):
            The axis to add the colorbar to. If None, use the current axis.
    Returns:
        cbar (mpl.colorbar.Colorbar):
            The colorbar object.
    '''
    # If the axis is None, use plt.gca()
    ax = ax or plt.gca()

    # Get the mappable from the axis
    mappable = ax.get_images()[0]
    if not isinstance(mappable, mpl.cm.ScalarMappable):
        raise ValueError('The axis does not contain a mappable.')
    
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size=width, pad=pad)

    fig = ax.figure
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
    cbar.set_label(label, rotation = 90, labelpad = labelpad, fontsize = fontsize)

    return cbar

def draw_colorbar_to_ax(vmin: float,
                        vmax: float,
                        cmap: mpl.colors.Colormap,
                        label: str = '',
                        labelpad: float = 8,
                        fontsize: int = 16,
                        cbar_ax: plt.Axes = None):
    '''
    Draws a colorbar to a given axis.
    Args:
        vmin (float):
            The minimum value for the colorbar.
        vmax (float):
            The maximum value for the colorbar.
        cmap (mpl.colors.Colormap):
            The colormap to use for the colorbar.
        label (str):
            The label for the colorbar.
        cbar_ax (plt.Axes):
            The axis to draw the colorbar to. If None, use the current axis.
    Returns:
        cbar (mpl.colorbar.Colorbar):
            The colorbar object.
    '''
    # If the axis is None, use plt.gca()
    cbar_ax = cbar_ax or plt.gca()

    # Set the colorbar properties
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    color_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = cbar_ax.figure
    cbar = fig.colorbar(color_mappable, cax=cbar_ax, orientation='vertical')
    cbar.set_label(label, rotation=90, labelpad = labelpad, fontsize = fontsize)

    return cbar
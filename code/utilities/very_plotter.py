import numpy as np
import wesanderson
import string


def get_fig_template(plt):
    """This function sets some plt defaults and returns blue and red color
    palettes

        Input
            plt     : Matplotlib instance

        Output
            plt     : update Matplotlib instance
            colors  : blue and red color palettes

    """

    # plt default parameters
    rcParams = {
        'text.usetex': 'True',
        'axes.spines.top': 'False',
        'axes.spines.right': 'False',
    }
    plt.rcParams.update(rcParams)

    # Define color palettes
    greens = [wesanderson.color_palettes['Castello Cavalcanti'][0][3],
              wesanderson.color_palettes['Castello Cavalcanti'][0][1]]

    blues = [wesanderson.color_palettes['Darjeeling Limited'][1][1],
             wesanderson.color_palettes['The Life Aquatic with Steve Zissou'][0][0],
             wesanderson.color_palettes['Darjeeling Limited'][1][3]]
    yellows = [wesanderson.color_palettes['Isle of Dogs'][1][2],
               wesanderson.color_palettes['Darjeeling Limited'][1][2],
               #wesanderson.color_palettes['Hotel Chevalier'][0][3],
               wesanderson.color_palettes['Darjeeling Limited'][1][0]]
    # blues = np.array([[7, 47, 95], [18, 97, 160], [56, 149, 211], [88, 204, 237]]) / 255  # shades of blue
    # reds = np.array([[167, 0, 0], [255, 82, 82], [255, 123, 123], [255, 186, 186]]) / 255  # shades of reds

    # from collections import OrderedDict
    # cmaps = OrderedDict()
    # cmaps['Sequential'] = [
    #     'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    #
    # greens = plt.get_cmap('YlGn')
    # blues = plt.get_cmap('PuBuGn')
    # yellows = plt.get_cmap('copper')
    #
    # greens_ = [greens(int(i)) for i in np.linspace(greens.N / 2, greens.N - (greens.N / 4), 2)]
    # blues_for_bar = [blues(int(i)) for i in np.linspace(0, blues.N, 3)]
    # yellows_for_bar = [yellows(int(i)) for i in np.linspace(0, yellows.N, 3)]

    return plt, greens, blues, yellows  # , blues, reds


def config_axes(ax, title, y_label, y_lim, xticks=None, xticklabels=None):
    """Set basic setting for plot axes"""
    ax.set_title(title, size=16)
    ax.grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(y_lim)
    ax.set_yticks(np.linspace(0, y_lim[1], 6))
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=12)


def plot_bar(ax, x, height, colors, bar_width=0.6, yerr=None):
    """Plot bars with error bar if given"""
    ax.bar(x=x, height=height, yerr=yerr,
                 width=bar_width, capsize=10,
                 color=colors, zorder=0)


def plot_bar_scatter(ax, data, color, bar_width):
    """Plot scatters over bar with half bar_width scatter range"""
    scatter_width = bar_width * (3/4)

    # Sort single data points to scatter
    unique, counts = np.unique(data, return_counts=True)
    y_counts_dic = dict(zip(unique, counts))
    max_y_number = max(y_counts_dic.values())
    y_x_pos = []
    y_values = []
    for y_value, y_count in y_counts_dic.items():
        if y_count == 1:
            positions = [0]
        else:
            positions = (np.linspace(0, (y_count * scatter_width / max_y_number), y_count)
                         - y_count * scatter_width / max_y_number / 2)
        y_x_pos.extend(positions)
        y_values.extend(y_count * [y_value])

    ax.scatter(y_x_pos, y_values, alpha=0.6, s=6, color=color, zorder=1)


def add_letters(ax):
    """Add letters to subplots"""
    for key, value in ax.items():
        value.text(-0.05, 1.1, string.ascii_lowercase[key],
                   transform=value.transAxes,
                   size=20, weight='bold')

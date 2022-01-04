import numpy as np
import wesanderson
import string
import palettable


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
        'yaxis.labellocation': 'bottom'
    }
    plt.rcParams.update(rcParams)

    # # Define color palettes from wesanderson
    col_exp = [wesanderson.color_palettes['Castello Cavalcanti'][0][3],
               wesanderson.color_palettes['Castello Cavalcanti'][0][1]]
    #
    # col_A = [wesanderson.color_palettes['Darjeeling Limited'][1][1],
    #          wesanderson.color_palettes['The Life Aquatic with Steve Zissou'][0][0],
    #          wesanderson.color_palettes['Darjeeling Limited'][1][3]]
    # col_C = [wesanderson.color_palettes['Isle of Dogs'][1][2],
    #            wesanderson.color_palettes['Darjeeling Limited'][1][2],
    #            #wesanderson.color_palettes['Hotel Chevalier'][0][3],
    #            wesanderson.color_palettes['Darjeeling Limited'][1][0]]
    # col_A = np.array([[7, 47, 95], [18, 97, 160], [56, 149, 211], [88, 204, 237]]) / 255  # shades of blue
    # reds = np.array([[167, 0, 0], [255, 82, 82], [255, 123, 123], [255, 186, 186]]) / 255  # shades of reds

    # from collections import OrderedDict
    # cmaps = OrderedDict()
    # cmaps['Sequential'] = [
    #     'Greys', 'Purples', 'col_A', 'col_exp', 'Oranges', 'Reds',
    #     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    #
    # col_exp = plt.get_cmap('YlGn')
    # col_A = plt.get_cmap('PuBuGn')
    # col_C = plt.get_cmap('copper')
    #
    # greens_ = [col_exp(int(i)) for i in np.linspace(col_exp.N / 2, col_exp.N - (col_exp.N / 4), 2)]
    # blues_for_bar = [col_A(int(i)) for i in np.linspace(0, col_A.N, 3)]
    # yellows_for_bar = [col_C(int(i)) for i in np.linspace(0, col_C.N, 3)]

    tab20 = plt.get_cmap('tab20')
    tab20b = plt.get_cmap('tab20b')
    tab20c = plt.get_cmap('tab20c')
    col_exp = [tab20c(11), tab20c(8)]
    col_A = [tab20c(0), tab20b(18), tab20(18)]
    col_C = [wesanderson.color_palettes['Darjeeling Limited'][1][0],
             wesanderson.color_palettes['Darjeeling Limited'][1][2],
             # wesanderson.color_palettes['Hotel Chevalier'][0][3],
             wesanderson.color_palettes['Isle of Dogs'][1][2]]

    curl20 = palettable.cmocean.diverging.Delta_20.colors
    viridis_20 = palettable.matplotlib.Viridis_20.colors
    magma_20 = palettable.matplotlib.Magma_20.colors
    # col_exp = [[value / 255 for value in list_] for list_ in [curl20[-3], curl20[-5]]]
    col_A = [[value / 255 for value in list_] for list_ in [curl20[2], curl20[8], curl20[6]]]

    col_exp = [[value / 255 for value in list_] for list_ in [viridis_20[4], viridis_20[1]]]
    col_A = [[value / 255 for value in list_] for list_ in [viridis_20[15], viridis_20[18], viridis_20[9]]]
    return plt, col_exp, col_A, col_C  # , col_A, reds


def config_axes(ax, y_label=None, y_lim=None, title=None, x_label=None, x_lim=None,
                xticks=None, xticklabels=None, yticks=None, ytickslabels=None):
    """Set basic setting for plot axes"""
    ax.grid(True, axis='y', linewidth=.3, color=[.9, .9, .9])
    if title is not None:
        ax.set_title(title, size=16)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=12, loc='center')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=12)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=11)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ytickslabels is not None:
        ax.set_yticklabels(ytickslabels, fontsize=11)


def plot_bar(ax, x, height, colors, bar_width=0.6, errorbar_size=10, yerr=None):
    """Plot bars with error bar if given"""
    ax.bar(x=x, height=height, yerr=yerr,
           width=bar_width,
           color=colors, zorder=0,
           clip_on=False,
           error_kw=dict(ecolor='gray', lw=2, capsize=errorbar_size, capthick=0.9, elinewidth=0.9))


def plot_bar_scatter(ax, data, color, bar_width):
    """Plot scatters over bar with half bar_width scatter range"""
    scatter_width = bar_width * (3 / 4)

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

    ax.scatter(y_x_pos, y_values, alpha=0.4, s=6, color=color, zorder=1, clip_on=False)


def add_letters(ax):
    """Add letters to subplots"""
    for key, value in ax.items():
        value.text(-0.05, 1.1, string.ascii_lowercase[key],
                   transform=value.transAxes,
                   size=22, weight='bold')

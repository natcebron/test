"""
    Graphs DeMACIA-RX pour Streamlit - Datasets biaisés
    ===================================================
    Version: 0.2.0
    Author: Matthieu PELINGRE
    Date: June 28, 2022
    Purpose:
    Importe les données stockées dans les fichiers pickle :
        ~/biaised_graphs_data/graph*.pickle

    Crée les objets figures bokeh pouvant être affichés via st.bokeh_chart

    Exemple:
        import graphs_bokeh
        graph1a = graphs_bokeh.main('graph1a', caption='Distribution de la luminosité (Radiographies complètes)')
        st.bokeh_chart(graph1a, use_container_width=True)

    Dependencies: pickle, numpy, itertools, bokeh==2.4.1, streamlit
"""

# ======================================================================================================================
# MODULES
# ======================================================================================================================
import pickle
import numpy as np
from bokeh.plotting import figure  # bokeh version 2.4.1 !!!
from bokeh.models import ColumnDataSource, Label, Span
from bokeh.palettes import Category10 as palette
import itertools
import streamlit as st

# ======================================================================================================================
# PALETTES DE COULEURS
# ======================================================================================================================
colors = itertools.cycle(palette[10])
full_color = itertools.cycle(('#17becf', '#1f77b4', '#ff7f0e'))
masked_color = itertools.cycle(('#ff7f0e', '#2ca02c', '#d62728'))
normup_color = itertools.cycle(('#7f7f7f', '#bcbd22', '#17becf'))
covdown_color = itertools.cycle(('#9467bd', '#8c564b', '#e377c2'))
masks_color = itertools.cycle(('#1f77b4', '#ff7f0e', '#2ca02c'))


# ======================================================================================================================
# FONCTIONS
# ======================================================================================================================
def bokeh_distrib(dict_set, dark_mode=False, colors=colors, caption='title'):
    p = figure(plot_width=390, plot_height=260)

    for class_name, color in zip(dict_set, colors):
        nb_imgs = len(dict_set[class_name])  # nombre d'images de chaques classes

        hist, bin_edges = np.histogram(dict_set[class_name], bins=np.arange(0, 256, 5))

        hist = hist / nb_imgs  # on normalise car nombre légèrement différent par classe

        source = ColumnDataSource({
            'hist': hist,
            'x': bin_edges[:-1]
        })

        hist1 = p.vbar(
            source=source,
            x='x',
            top='hist',
            width=bin_edges[1] - bin_edges[0],
            color=color,
            fill_alpha=.5,
            legend_label=class_name
        )

        # ligne médianne
        median_class = np.median(dict_set[class_name])

        s = Span(
            dimension='height',
            location=median_class,
            line_color=color,
            line_width=3)

        p.add_layout(s)

        text_median = Label(
            x=median_class,
            y=np.max(hist) * 1.02,
            x_offset=5,
            text=f"{median_class:.0f}",
            text_color=color,
            text_font_size="6pt"
        )

        p.add_layout(text_median)

    p.legend.click_policy = 'hide'
    p.legend.label_text_font_size = "6pt"

    p.title = caption
    p.title.align = 'center'
    p.title.text_font_size = '12px'
    p.title_location = 'below'

    p.axis.major_label_text_font_size = '7px'

    if dark_mode:
        p.background_fill_color = '#0E1117'
        p.border_fill_color = '#0E1117'
        p.axis.axis_line_color = '#eeeeee'
        p.axis.major_label_text_color = '#eeeeee'
        p.legend.background_fill_color = '#0E1117'
        p.legend.label_text_color = '#eeeeee'
        p.grid.grid_line_color = '#5b5b5b'
        p.title.text_color = '#9b9b9b'

    return p


def main(graph, caption='title'):

    if (graph == 'graph2a') or (graph == 'graph3a'):
        with open(f'data/graph2&3a.pickle', 'rb') as f:
            graph_dict = pickle.load(f)
    else:
        with open(f'data/{graph}.pickle', 'rb') as f:
            graph_dict = pickle.load(f)

    if graph == 'graph1a':
        graph_colors = full_color
    elif graph == 'graph2b':
        graph_colors = normup_color
    elif graph == 'graph2c':
        graph_colors = covdown_color
    elif graph == 'graph3b':
        graph_colors = masks_color
    else:
        graph_colors = masked_color

    return bokeh_distrib(graph_dict, dark_mode=True, colors=graph_colors, caption=caption)


if __name__ == '__main__':  # test
    st.bokeh_chart(main('graph3b'), use_container_width=True)

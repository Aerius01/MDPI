import matplotlib.pyplot as plt
import numpy as np
import os

def setup_plot_aesthetics(ax, title, xlabel='Concentration (ind*L⁻¹)', ylabel='Depth (m)', title_fontsize=30, label_fontsize=30, tick_fontsize=30):
    """Sets labels, title, and font sizes for a plot."""
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

def configure_axes(ax, max_depth, max_concentration, is_symmetric=False, depth_tick_step=1, conc_tick_step=100):
    """Configures the axes for the profile plot."""
    # Y-axis (Depth)
    safe_max_depth = 0 if np.isnan(max_depth) else max_depth
    max_y_axis = np.ceil(safe_max_depth)
    y_ticks = np.arange(0, max_y_axis + depth_tick_step, depth_tick_step)
    ax.set_yticks(y_ticks)
    ax.set_ylim(max(y_ticks) if len(y_ticks) > 0 else 1, 0)

    # X-axis (Concentration)
    safe_max_concentration = 0 if np.isnan(max_concentration) else max_concentration
    if safe_max_concentration == 0:
        safe_max_concentration = conc_tick_step
    
    current_max_conc = np.ceil(safe_max_concentration)

    if is_symmetric:
        x_limit = (-current_max_conc, current_max_conc)
        x_ticks = np.arange(-current_max_conc, current_max_conc + conc_tick_step, conc_tick_step)
        x_labels = [abs(int(x)) for x in x_ticks]
    else:
        x_limit = (0, current_max_conc)
        x_ticks = np.arange(0, current_max_conc + conc_tick_step, conc_tick_step)
        x_labels = [int(x) for x in x_ticks]

    ax.set_xlim(x_limit)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

def save_plot(fig, output_path, plot_title):
    """Saves the plot to the specified path."""
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f'{plot_title}.png'), dpi=300)
    plt.close(fig) 
import matplotlib.pyplot as plt
import numpy as np
import os

def setup_plot_aesthetics(ax, title, xlabel='Concentration (ind*L⁻¹)', ylabel='Depth (m)', title_fontsize=30, label_fontsize=30, tick_fontsize=30):
    """Sets labels, title, and font sizes for a plot."""
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

def _calculate_nice_step(range_span: float, max_ticks: int) -> float:
    """Returns a 'nice' tick step (1/2/5 * 10^n) so the number of ticks
    stays under max_ticks for the given range_span."""
    if range_span <= 0:
        return 1
    raw_step = max(range_span / max(1, max_ticks), 1e-6)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1:
        nice = 1
    elif residual <= 2:
        nice = 2
    elif residual <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def configure_axes(ax, max_depth, max_concentration, is_symmetric=False, depth_tick_step=1, conc_tick_step=100, max_conc_ticks=8):
    """Configures the axes for the profile plot.
    Limits x-axis tick count to avoid overlapping labels.
    """
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

    # Determine an effective step that keeps tick count reasonable
    range_span = 2 * current_max_conc if is_symmetric else current_max_conc
    approx_ticks = range_span / max(1, conc_tick_step)
    if approx_ticks > max_conc_ticks:
        conc_tick_step = _calculate_nice_step(range_span, max_conc_ticks)

    # Determine decimal precision for tick labels based on step
    if conc_tick_step >= 1:
        decimals = 0
    elif conc_tick_step >= 0.1:
        decimals = 1
    elif conc_tick_step >= 0.01:
        decimals = 2
    else:
        decimals = 3

    def _fmt(val: float) -> str:
        s = f"{val:.{decimals}f}"
        # Trim trailing zeros and dot for cleaner labels
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    if is_symmetric:
        x_limit = (-current_max_conc, current_max_conc)
        x_ticks = np.arange(-current_max_conc, current_max_conc + conc_tick_step, conc_tick_step)
        x_labels = [_fmt(abs(x)) for x in x_ticks]
    else:
        x_limit = (0, current_max_conc)
        x_ticks = np.arange(0, current_max_conc + conc_tick_step, conc_tick_step)
        x_labels = [_fmt(x) for x in x_ticks]

    # Final safety: limit the number of ticks to avoid label overlap
    max_allowed = max_conc_ticks + 1  # include 0 and max
    if len(x_ticks) > max_allowed and max_allowed > 1:
        step_idx = int(np.ceil(len(x_ticks) / max_allowed))
        idxs = np.arange(0, len(x_ticks), step_idx)
        # Ensure last tick is included
        if idxs[-1] != len(x_ticks) - 1:
            idxs = np.append(idxs, len(x_ticks) - 1)
        x_ticks = x_ticks[idxs]
        x_labels = [x_labels[i] for i in idxs]

    ax.set_xlim(x_limit)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

def save_plot(fig, output_path, file_name):
    """Saves the plot to the specified path."""
    os.makedirs(output_path, exist_ok=True)
    # Tight layout reduces label/title overlap and clipping
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f'{file_name}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
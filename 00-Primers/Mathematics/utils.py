
import matplotlib.pyplot as plt
import numpy as np

import warnings

axis_arrow_params = {
  'width': 0.025,
  'length_includes_head': True,
  'head_width': 0.2,
  'head_length': None,
  'shape': 'full',
  'overhang': 0,
  'head_starts_at_zero': True,
  #
  'color': 'grey',
  'linestyle': '--'
}

text_params = {
  'fontsize': 'x-large',
  # 'horizontalalignment': 'right',
  # 'verticalalignment': 'top',
  'weight': 'bold'
}

def make_grid(xlim=None, ylim=None, no_origin=False, silent=False, **kwargs):
  """Creates a grid for plotting

  Args:
    xlim, ylim: The limits of the environment
    no_origin: Skips drawing the origin text
    silent: Suppresses all texts
  """
  figsize = kwargs.pop('figsize', (6, 6))
  fig = plt.figure(figsize=figsize, frameon=False)
  if xlim is None:
    xlim = [-4, 4]
  if ylim is None:
    ylim = [-4, 4]
  plt.xlim(xlim)
  plt.ylim(ylim)
  # Make arrows:
  plt.arrow(xlim[0], 0, xlim[1] - xlim[0], 0, **axis_arrow_params)
  plt.arrow(0, ylim[0], 0, ylim[1] - ylim[0], **axis_arrow_params)
  if not silent:
    plt.text(xlim[1], 0, 'x', ha='left', va='center', **text_params)
    plt.text(0, ylim[1], 'y', ha='center', va='bottom', **text_params)

    if not no_origin:
      plt.text(0, 0, '(0, 0)', ha='right', va='top', **text_params)

  plt.grid('both')
  fig.silent = silent
  return fig

def add_vector(x, y=None, origin=None, name=None, line_text=None, **kwargs):
  """Adds a vector to an existing grid.

  Args:
    x: Either x-coordinate or a (x, y) point.
    y: Either y-coordinate or None. If None, the first argument is assumed to
       be a point
    origin: Origin of the vector, if None defaults to (0, 0)
    name: Name of the vector.
    line_text: Any additional text to be added to the line
    **kwargs: All other keyword arguments are passed to the `plt.arrow`.
  """
  if y is None:
    while x.ndim > 1:
      x = x[0]
    x, y = x[0], x[1]
  color = next(plt.gca()._get_lines.prop_cycler)['color']
  delta_x = 0.2 * np.sign(x)
  delta_y = 0.2 * np.sign(y)
  if origin is None:
    Ox, Oy = 0, 0
  else:
    Ox, Oy = origin
  arr = plt.arrow(Ox, Oy, x, y, width=0.05, head_width=0.2, color=color, **kwargs)

  fig = plt.gcf()
  if hasattr(fig, 'silent') and fig.silent:
    return arr

  text = f'({x}, {y})'
  if name is not None:
    text = name + text

  plt.text(x+delta_x, y+delta_y, text,
           ha='left' if x>=0 else 'right',
           va='top' if y<=0 else 'bottom',
           color=color,
           **text_params)
  if line_text is not None:
    theta = np.arctan2(y, x) / np.pi * 180
    if theta > 90 or theta < -270:
      theta += 180
    # theta = 30
    plt.text(x / 2, y / 2, line_text, ha='center', va='bottom',
             rotation=theta, rotation_mode='anchor',
             fontsize='large', weight='bold')
  return arr

def add_line(point_a, point_b, color=None, **kwargs):
  """Draws a line in the active axes.

  Args:
    point_a, point_b: tuple of (x, y) coordinates of the points the describe
                      the line.
    color: line color. If the color is another plot,
           the color is extracted from it
    **kwargs: Keyword arguments are passed to the `plt.plot`
  """
  if color is None:
    color = next(plt.gca()._get_lines.prop_cycler)['color']
  elif hasattr(color, 'get_color'):
    color = color.get_color()
  elif hasattr(color, 'get_facecolor'):
    color = color.get_facecolor()
  elif hasattr(color, 'get_edgecolor'):
    color = color.get_edgecolor()

  linestyle = kwargs.pop('linestyle', '--')

  dy = point_b[1] - point_a[1]
  dx = point_b[0] - point_a[0]
  ax = plt.gca()

  x0, x1 = ax.get_xlim()
  y0, y1 = ax.get_ylim()

  if dy == 0 and dx == 0:
    raise ValueError('Cannot plot a line through a single point')
  if dy == 0:
    y0 = y1 = point_a[1]
  elif dx == 0:
    x0 = x1 = point_a[0]
  else:
    m = dy / dx
    b = point_a[1] - point_a[0] * m
    y0 = x0 * m + b
    y1 = x1 * m + b

  line = plt.plot([x0, x1], [y0, y1], c=color, ls=linestyle, **kwargs)

  return line

def add_segment(point_a, point_b, color=None, **kwargs):
  """Draws a line in the active axes.

  Args:
    point_a, point_b: tuple of (x, y) coordinates of the points the describe
                      the line.
    color: line color. If the color is another plot,
           the color is extracted from it
    **kwargs: Keyword arguments are passed to the `plt.plot`
  """
  if color is None:
    color = next(plt.gca()._get_lines.prop_cycler)['color']
  elif hasattr(color, 'get_color'):
    color = color.get_color()
  elif hasattr(color, 'get_facecolor'):
    color = color.get_facecolor()
  elif hasattr(color, 'get_edgecolor'):
    color = color.get_edgecolor()

  linestyle = kwargs.pop('linestyle', '-')
  marker = kwargs.pop('marker', '|')

  x0, y0 = point_a
  x1, y1 = point_b

  line = plt.plot([x0, x1], [y0, y1], color=color, linestyle=linestyle, **kwargs)

  return line

import matplotlib as mpl
import supports
import inspect

class DrawFigure:
    def __init__(self, figure):
        self.__figure = figure
        self.__axes = figure.gca()
        self.scalar = 1
        self.__internal_scalar = 1

    def add_cylinder(self, x, y, h, w, color, ctype='filled', elp=.7):
        """Method that adds a cylinder to the figure.
        :param x: the x position of the cylinder.
        :param y: the y position of the cylinder.
        :param h: the height of the cylinder.
        :param w: the width of the cylinder.
        :param color: the color of the cylinder.
        :param ctype: the type of cylinder. If 'filled' the cylinder will be dense. If 'hollow' the cylinder will be
        :param elp: the ellipticity of the cylinder top and bottom.
        hollow. The default is 'filled'."""

        # apply internal scalar
        h *= self.__internal_scalar
        w *= self.__internal_scalar

        _ch = h - w * elp  # set cylinder height before scaling ellipse width
        _eh = w * elp  # set ellipse height relative to the width
        _sw = w * self.scalar  # scale width to the axes

        bottom_ellipse = mpl.patches.Ellipse((x, y), _sw, _eh, lw=0, facecolor=color, fill=True)
        rectangle = mpl.patches.Rectangle((x - _sw / 2, y), _sw, _ch, facecolor=color, fill=True)
        top_ellipse = mpl.patches.Ellipse((x, y + h - _eh), _sw, _eh, lw=0, facecolor=supports.highlight(color, -40),
                                          fill=True)
        self.__figure.gca().add_patch(bottom_ellipse)
        self.__figure.gca().add_patch(rectangle)
        self.__figure.gca().add_patch(top_ellipse)

        if ctype == 'hollow':
            _heh = _eh * .7  # set insert ellipse height
            _hew = _sw * .7
            ins_ellipse = mpl.patches.Ellipse((x, y + h - _eh), _hew, _heh, lw=0,
                                              facecolor=supports.highlight(color, 10), fill=True)
            self.__figure.gca().add_patch(ins_ellipse)

    def get_figure_scalar(self):
        """Method that grabs figure finds the w/h ratio of the passed figure. This should be run before any figures
        are drawn to ensure they have the correct proportions."""
        x_lim, y_lim = self.__figure.gca().get_xlim(), self.__figure.gca().get_ylim()
        plot_dims = self.__figure.gca().get_window_extent()
        self.scalar = (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0]) * plot_dims.height / plot_dims.width
        self.__internal_scalar = (y_lim[1] - y_lim[0]) / 2.466

    def add_circle(self, x, y, w, color, ctype='filled'):
        w *= self.__internal_scalar
        _sw = w * self.scalar
        circle = mpl.patches.Ellipse((x, y), _sw, w , lw=0, facecolor=color, fill=True)
        self.__figure.gca().add_patch(circle)
        if ctype == 'hollow':
            _heh = w * .6
            _hew = _sw * .6
            ins_circle = mpl.patches.Ellipse((x, y), _hew, _heh, lw=0,
                                              facecolor=supports.highlight(color, 50), fill=True)
            self.__figure.gca().add_patch(ins_circle)

    def grid(self, function, grid, spacing, gtype='sqr', depth=False, **kwargs):

        # handle all kwargs
        _sw = kwargs['w'] * self.scalar * self.__internal_scalar
        _w = kwargs['w'] * self.__internal_scalar
        _x, _y, _c = kwargs['x'], kwargs['y'], kwargs['color']  # store initial values to iterate upon
        if 'elp' not in kwargs:
            kwargs['elp'] = 1
        function_args = {}
        for k in kwargs:  # prevent non-passable kwargs from making it to the function
            if k in inspect.getfullargspec(function)[0]:
                function_args[k] = kwargs[k]

        _h = 0  # set highlight shift
        x_shift, y_shift = 0, 0
        x_range = grid[0]
        for yid in range(grid[1]):
            function_args['color'] = supports.highlight(kwargs['color'], _h)  # set a color to use
            y_shift = -_w * yid * (1 + spacing) * kwargs['elp']
            if gtype == 'sqr':
                if depth is True:
                    x_shift = -_sw * yid * spacing
                else:
                    x_shift = 0
            elif gtype == 'hex':
                if yid % 2 == 0:  # if current cylinder x-index is even, cut one cylinder from the row
                    if depth is True:
                        x_shift = -_sw * yid * spacing
                    else:
                        x_shift = 0
                    x_range = grid[0] - 1
                else:  # if current cylinder index is uneven, shift the row differently compared to normal
                    if depth is True:
                        x_shift = -_sw * yid * (1 - spacing)
                    else:
                        x_shift = -(_sw * (1 + spacing)) / 2
                    x_range = grid[0]
            for _ in range(x_range):
                function_args['x'] = kwargs['x'] +  x_shift
                function_args['y'] = kwargs['y'] + y_shift  # update xy before passing
                function(**function_args)  # add cylinder to figure
                x_shift += _sw * (1 + spacing)
            if depth is True:
                _h += 15  # update highlight shift

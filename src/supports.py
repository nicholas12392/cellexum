import threading
import os
import json
import collections
import numpy as np
import time
import cv2


# ATTRIBUTES
__cwd__ = os.path.dirname(os.path.realpath(__file__))  # grab working directory
__cache__ = rf'{__cwd__}\__cache__'  # add cache path
__gpx__ = r'{}\graphics\raster'.format(__cwd__.removesuffix(r'\src'))

__cp__ = {'bg': '#474747',
          'fg': '#ffffff',
          'disabled_bg': '#707070',
          'disabled_fg': 'darkgray',
          'base': '#476cda',
          'dark_base': '#2a4185',
          'light_base': '#6688eb',
          'dark_bg': '#2b2b2b',
          'dark_fg': '#1c1c1c'}  # color profile definition
button_font = 'Arial 12 bold'

with open(rf'{__cwd__}\__misc__\tooltips.json', 'r') as f:
    __tt__ = json.load(f)  # load-in tooltips

with open(rf'{__cwd__}\__misc__\warnings.json', 'r') as f:
    __warnings__ = json.load(f)  # load-in warnings

# CLASSES
class NumpyArrayEncoder(json.JSONEncoder):
    """Source: https://pynative.com/python-serialize-numpy-ndarray-into-json/"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ColorGradient:
    """Class that allows for creating a true color gradient."""
    def __init__(self, onset=None, end=None, out='hex'):
        if onset[0] == '#':
            onset = hex_to_rgb(onset)
        if end[0] == '#':
            end = hex_to_rgb(end)

        self.onset = onset
        self.end = end
        self.out = out

    def gradient(self, state):
        _0 = np.power(self.onset, 2)
        _1 = np.power(self.end, 2)

        color = []  # calculate each color channel for the gradient
        for c0, c1 in zip(_0, _1):
            _a = (c0 - c1) / -1
            _b = c0
            color.append(int(np.round(_a * state + _b, 0)))

        rgb = np.sqrt(np.array(color))
        if self.out == 'hex':
            return rgb_to_hex(rgb)
        else:
            return rgb


# DECORATORS
def thread_daemon(func):
    def wrapper(self, *args, **kwargs):
        threading.Thread(target=func, args=(self, *args), kwargs=kwargs, daemon=True).start()
    return wrapper


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        diff = end - start
        tprint(f"Time elapsed for {func.__name__}: {diff:.2f} s")
        return result
    return wrapper


def thread_main(func):
    def wrapper(self, *args, **kwargs):
        threading.Thread(target=func, args=(self, *args), kwargs=kwargs, daemon=False).start()
    return wrapper


# SUPPORT FUNCTIONS
def post_cache(params: dict, behavior='update'):
    """Wrapper function that pushes a dict to the settings.json cache file.
    :param params: the dict to be pushed.
    :param behavior: the behavior of json_dict_push to use when pushing the dict."""
    path = rf'{__cwd__}\__cache__\settings.json'
    json_dict_push(path, params=params, behavior=behavior)


def json_dict_push(path, params=None, behavior='update'):
    """
    Push json file to disk and overwrite existing variables with new values.
    :param path: Path to json file to be pushed.
    :param params: Parameters to be pushed. Dictionary of parameters. Default is None.
    :param behavior: Behavior of the script. Can be 'update', 'replace', 'mutate', 'read', or 'clear'. Default is
    'update'. 'update' will overwrite only existing variables. 'replace' will overwrite the file itself. 'mutate' will
    overwrite parent entries if they exist in the dict, and add the rest. 'read' will only read the file. 'clear' will
    clear the file content.
    :return: if behaviour is 'read', returns dictionary or None.
    """

    if not os.path.isfile(path):  # write file if it does not exist
        if behavior != 'read':
            with open(path, 'w') as file:
                file.write(json.dumps(params, indent=4, cls=NumpyArrayEncoder))  # write dict to json file
                file.truncate()  # truncate the file before closing
        else:  # return none if read is prompted on a non-existing file
            return {}  # consider making this an option between None and {}

    with open(path, 'r+') as file:
        data = json.load(file)  # load json file

        if behavior != 'read':
            file.seek(0)  # reset reference point in loaded file content
            if behavior == 'update':  # update existing dict
                json_dict = dict_update(data, params)
            elif behavior == 'replace':  # replace existing dict
                json_dict = params
            elif behavior == 'mutate':  # replace each parent entry in the existing dict
                for k, v in params.items():
                    data[k] = v
                json_dict = data
            elif behavior == 'clear':  # clear content of existing file
                json_dict = {}
            else:
                raise KeyError(f'Invalid behavior: {behavior!r}')
            file.write(json.dumps(json_dict, indent=4, cls=NumpyArrayEncoder))  # write dict to json file
            file.truncate()  # truncate the file before closing
        else:  # return existing dict
            return data


def dict_update(d, u):
    """Source: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def setting_cache(*args, clear_entry=False, replace=None):
    """Function that fetches an entry in the cache settings.json file specified by the arguments.
    :param args: dict keys from highest to lowest order to the desired dict entry.
    :param clear_entry: whether to clear the cache entry.
    :param replace: replacement for the value at the specified entry."""
    path = rf'{__cwd__}\__cache__\settings.json'
    cache = json_dict_push(path, behavior='read')
    if len(args) == 0:  # if no args exist in the call, return the entire cache
        return cache

    v, t = dict_trail(cache, *args, _vd=replace)
    if clear_entry is True:
        json_dict_push(path, params=t, behavior='update')
    return v


def dict_trail(d, *args, _vd=None, _i=0):
    """Function that fetches the value along with its trail in the form of a dictionary for argument keys leading to
    the specific entry in the input dict.
    :param d: input dictionary to trail.
    :param args: dict keys from highest to lowest order to the desired dict entry.
    :param _vd: internal entry.
    :param _i: internal entry."""
    if len(args) == 1:  # catch non-iterable single trails
        value, trail = d[args[0]], {args[0]: _vd}
        return value, trail
    for _ in args:
        if len(args) > _i:
            dict_trail(d[args[_i]], *args, _vd={list(reversed(args))[_i]: _vd}, _i=_i + 1)
        else:
            value, trail = d, _vd
            return value, trail


def tprint(text) -> None:
    """
    Print the time in front of the printed message.
    :param text: The text to be printed.
    :return: Prints the text along with the current time.
    """
    time_stamp = time.localtime()
    print(f'{time.strftime("%H:%M", time_stamp)} - {text}')


def hex_to_rgb(hex: str) -> tuple[int, ...]:
    """Converts a hex string to RGB tuple."""
    _rgb = tuple(int(hex.removeprefix('#')[i:i + 2], 16) for i in (0, 2, 4))
    return _rgb


def rgb_to_hex(rgb) -> str:
    """Converts an rgb color tuple to a hex color code."""
    r, g, b = [int(i * 255) if not isinstance(i, int) else i for i in rgb]  # scale the color input
    _hex = f'#{r:02x}{g:02x}{b:02x}'
    return _hex


def highlight(color, percent=20):
    """Brightens a color with a given percent. If the highlight exceeds 255 on any channel, the output channel is
    normalized to 255."""

    hex_type = False
    if isinstance(color, str):  # if color is str, assume hex code, otherwise assume rgb
        color = hex_to_rgb(color)
        hex_type = True

    highlight_color = []
    scalar = 1 + percent / 100
    for c in color:
        _hc = int(c * scalar)  # determine new channel intensity
        if _hc > 255:  # fix top limit
            _hc = 255
        elif _hc < 0:  # fix bottom limit
            _hc = 0
        highlight_color.append(_hc)

    if hex_type is True:  #
        highlight_color = rgb_to_hex(highlight_color)

    return highlight_color


class ColorPresets:
    def __init__(self, default=None):
        self._presets = json_dict_push(rf'{__cwd__}\__misc__\colors.json', behavior='read')
        self.default = default

    def get(self, n, preset=None, hex=False):
        if self.default is not None and preset is None:
            colors = self.generate(n, **self._presets[self.default])
        elif preset is not None:
            colors = self.generate(n, **self._presets[preset])
        else:
            raise ValueError('No preset specified.')

        if hex is True:
            colors = tuple(map(rgb_to_hex, colors))
        return colors

    @property
    def presets(self):
        return list(self._presets.keys())

    @staticmethod
    def generate(n, tt, lin, gauge, freq, boldness) -> tuple[tuple[any], ...]:
        """
        Generates an array of colors from an RGB spectrum through the oscillator line function. Function is a stripped
        version of that found in the nanoscipy package.
        :param n: Number of colors to generate.
        :param tt: The trigonometry type for the R, G, and B channels. Can be 'sin' or 'cos'.
        :param lin: Linearity for the R, G, and B channels.
        :param gauge: Maximum amplitude span of the oscillator for the R, G, and B channels.
        :param freq: Frequency of the oscillator for the R, G, and B channels.
        :param boldness: Introduces a minimum baseline for the R, G, and B channels. A higher value streamlines the color
        spectrum, such that a more easily pleasing spectrum is obtained. A lower value makes the colors in the spectrum
        more easily distinguishable.
        generated spectrum. Default is False.
        :return: A numpy array with the color spectrum points in the form ((R,G,B), ...).
        """

        channels = []
        for i in range(3):
            channels.append(ColorPresets.osc_line(n, tt[i], lin[i], gauge[i], freq[i], boldness))

        return tuple(zip(*channels))

    @staticmethod
    def osc_line(points, tt, lin, gauge, freq, min_shift):
        """
        Creates an oscillator line with given parameters. Function is overall the same as the version found in the
        nanoscipy package.
        :param points: The amount of points there should be on the line.
        :param tt: The trigonometry type. Can be 'sin' or 'cos'. Default is 'sin'.
        :param lin: The linearity. Default is 1.
        :param gauge: Maximum amplitude span of the oscillator. Default is 0.5.
        :param freq: The frequency of the oscillator. Default is 4.
        :param min_shift: A minimum shift of the oscillator baseline. Default is 0.1.
        :return: A numpy array with the oscillator line points.
        """
        lrange = np.linspace(0, 1, points)  # create the amount of desired points
        y1, y2 = 0, lin
        lr = ((y1 - y2) / -1, 1 - lin)  # find slope
        lin_values = lr[0] * lrange + lr[1]  # define linear values

        # find trigonometric values
        if tt == 'sin':
            trig_values = gauge / 2 * np.sin(lrange * freq * np.pi) + 0.5
        elif tt == 'cos':
            trig_values = gauge / 2 * np.cos(lrange * freq * np.pi) + gauge / 2
        else:
            raise ValueError('tt must be sin/cos')

        # adjust baseline values and return the oscillator line points
        func_max = 1 + max(trig_values)
        func_min = min(lin_values + trig_values)
        return (lin_values + trig_values - func_min + min_shift) / (func_max - func_min + min_shift)


def colorize(img, hex):
    """
    Converts a black rgba image to have color.
    :param img: rgba image from OpenCV
    :param hex: hex-code color for the image
    :return: Colored image
    """

    relative_rgb = tuple(int(hex.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))  # convert color to rgb
    r, g, b, a = cv2.split(img)  # split image channels

    handled_channels = []
    for channel, color in zip((b, g, r), np.flip(relative_rgb)):
        c = channel - 255 * -1  # construct inversion matrix
        handled_channels.append(np.uint8(c * color))  # color inversion matrix
    _ = cv2.merge(handled_channels + [a])  # create new image matrix

    return _


def get_max_cpu():
    """Get the maximum CPU usage from the defined application settings."""
    defaults = json_dict_push(rf'{__cache__}\application.json', behavior='read')['ApplicationSettings']
    if defaults['MaxAbsoluteCPU'] != '':
        return defaults['MaxAbsoluteCPU']
    else:
        return int(os.cpu_count() * defaults['MaxRelativeCPU'])


def convert(value: str, t: type, debug: bool=False) -> str | int | float | None:
    """Convert a string to an int or a float. If the operation yields a TypeError, either debug or return the input.
    :param value: The string to convert.
    :param t: Type to convert to, either int or float.
    :param debug: Wheter a failed conversion should be debugged. Default is False."""
    try:
        return t(value)
    except TypeError:
        if debug:
            tprint('Failed to convert {} to {}'.format(value, t))
        else:
            return value
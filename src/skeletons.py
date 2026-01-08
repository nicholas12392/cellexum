import threading
import tkinter as tk
import base
from tkinter import filedialog, messagebox, simpledialog
import os
import json
from PIL import Image, ImageTk, ImageOps, ImageFile
import supports
import concurrent.futures
import multiprocessing
import time
import cv2
from functools import partial
from supports import lprint as print

ImageFile.LOAD_TRUNCATED_IMAGES = True  # this allows for loading over SyntaxError: broken PNG file

row_counter = {}
grid_counter = {}


def auto_row(p, correction=None, shift=None):
    """Function that automatically places widget in the next row. 'p' is the row set. Stalling row update is done by
    placing a '@' in front of 'p'."""

    # start by checking whether row update should be stalled
    if p.startswith('@'):
        _check_stale = True
        p = p.replace('@', '')
    else:
        _check_stale = False

    if p in row_counter.keys():
        if not _check_stale:  # do not update the row if stale is True
            row_counter[p] += 1
    else:
        row_counter[p] = 0

    if correction is not None:  # correct the row if prompted
        row_counter[p] += correction

    if shift is not None:  # shift returned row if prompted
        return row_counter[p] + shift

    return row_counter[p]


class CommandWrapper:
    """Class that packs functions into a single call. The purpose is to group function calls for tkinter traces to
    allow what is effectively multi-tracing for a single tkinter variable."""

    def __init__(self):
        self.callbacks = {}
        self.cbname = None
        self.mode = None

    def __call__(self, *_):
        for callback in self.callbacks.values():
            callback()

    def __add__(self, other):
        """Add callback and grab a semi-unique id to remember it by."""
        self.callbacks[other.__qualname__] = other

    def __sub__(self, other):
        """Remove callback by id."""
        self.callbacks.pop(other.__qualname__)


class AppWidget(tk.Widget):
    """Wrapper for tkinter widget high-level functionality"""

    def __init__(self, parent, *args, **kwargs):
        if 'bg' not in kwargs and 'background' not in kwargs:
            kwargs['bg'] = supports.__cp__['bg']
        super().__init__(parent, *args, **kwargs)

        self.tkID = None  # allow for setting of unique tkinter ids
        self._all_binds = {}
        self._individual_binds = {}

        # set up protected tether attributes
        self.__tether = {}
        self.__tether_cache = {}
        self.__tether_activity = False
        self.__trigger_activity = False
        self.__internal_data = {'Cache': {}, 'Traces': set()}

        # accessible attributes to alter for tags and groups used by WindowFrame
        self.tag = None
        self.group = None

        self.bind('<Button-1>', self.post_self_click, '+')

    def grid(self, *args, **kwargs):

        n_row = None
        if 'row' in kwargs:  # overwrite all autorow features if row is passed to the grid
            n_row = kwargs['row']
            del kwargs['row']
            if 'autorow' in kwargs:
                del kwargs['autorow']

        if 'autorow' in kwargs:
            if 'arc' in kwargs:  # arc -> auto row correction
                n_row = auto_row(kwargs['autorow'], correction=kwargs['arc'])
                del kwargs['arc']
            elif 'ars' in kwargs:  # arc -> auto row shift
                n_row = auto_row(kwargs['autorow'], shift=kwargs['ars'])
                del kwargs['ars']
            else:
                n_row = auto_row(kwargs['autorow'])
            del kwargs['autorow']

        if n_row is not None:
            super().grid(row=n_row, *args, **kwargs)
        else:
            super().grid(*args, **kwargs)

    def tether(self, target, trigger, action=None, mode='=='):
        """Tethers self to a specific tkinter variable, which may trigger an action setting of a specific value.
        :param target: tkinter variable
        :param trigger: trigger value
        :param action: dictionary, containing arguments, kwargs, and potentially a selection that should be set when
        triggered.
        :param mode: trigger behaviour. Can be '!=' or '=='. The default is '=='.
        """

        self.__internal_data['Tether'] = {'target': target, 'trigger': trigger, 'action': action, 'mode': mode,
                                          'on_action': [], 'on_reset': []}
        self.__internal_data['Cache']['Tether'] = {}  # set up tether cache

        target.trace_add('write', self.__tether_check)
        self.__tether_check()  # catch trigger from default selection

    def __tether_check(self, *_):
        """Internal method that checks wether a target change should trigger tether action."""
        if self.__internal_data['Tether']['mode'] == '==':
            if self.__internal_data['Tether']['target'].get() == self.__internal_data['Tether']['trigger']:
                self.__tether_action('Tether')
            else:
                self.__tether_reset('Tether')
        elif self.__internal_data['Tether']['mode'] == '!=':
            if self.__internal_data['Tether']['target'].get() != self.__internal_data['Tether']['trigger']:
                self.__tether_action('Tether')
            else:
                self.__tether_reset('Tether')
        else:
            raise ValueError('Mode {!r} is invalid.'.format(self.__internal_data['Tether']['mode']))

    def __tether_action(self, bind):
        """Internal method that pushes tether action."""
        if self.__tether_activity is False:
            for k, v in self.__internal_data[bind]['action'].items():  # run initial tether action
                if k == 'selection':
                    self.__internal_data['Cache']['Tether'][k] = self.get()
                    self.set(v)
                else:
                    self.__internal_data['Cache']['Tether'][k] = self[k]
                    self[k] = v
            for f in self.__internal_data['Tether']['on_action']:  # run additional action functions
                f()
            self.__tether_activity = True

    def __tether_reset(self, bind):
        """Internal method that resets actions from tether action."""
        # if tether was activated, reset now from target-specific tether cache
        if self.__tether_activity is True:
            for k, v in self.__internal_data['Cache'][bind].items():  # run tether reset
                if k == 'selection':
                    self.set(v)
                else:
                    self[k] = v
            for f in self.__internal_data[bind]['on_reset']:  # run additional reset functions
                f()
            self.__tether_activity = False

    def trigger_tether(self, action=None):
        if action is None:  # preset action if none is passed
            action = self.tether_action

        if self.__tether_activity is False:
            self.__internal_data['Trigger'] = {'action': action}
            self.__internal_data['Cache']['Tether'] = {}  # set up tether cache
            self.__tether_action('Trigger')
            self.__tether_activity = True
        else:
            self.__tether_reset('Tether')
            self.__tether_activity = False

    def on_tether(self, function, on='action'):
        """Method for adding a function to a triggered tether. A tether must be defined before calling on_tether()."""

        if on == 'action':
            self.__internal_data['Tether']['on_action'].append(function)
        elif on == 'reset':
            self.__internal_data['Tether']['on_reset'].append(function)
        elif on == 'both':
            self.__internal_data['Tether']['on_action'].append(function)
            self.__internal_data['Tether']['on_reset'].append(function)
        else:
            raise ValueError(f'On method {on!r} is invalid.')

    def get_tethered(self, var=None):
        """Method for obtaining the information contained in a tether."""

        if var is None:
            return self.__internal_data['Tether']
        else:
            return self.__internal_data['Tether'][var]

    def manipulate_tether(self, entry, value=None):
        """Method that allows for downstream methods to manipulate a defined tether by directly changing the tether cache.
        :param entry: Can be dict or a str. If dict, all items in the dict is injected to the tether cache. If str, the
        str is used to inject that key with an associated value into the cache.
        :param value: If the entry is a str, this is the injected value into the tether cache."""
        if isinstance(entry, dict):
            for k, v in entry.items():
                self.__internal_data['Cache']['Tether'][k] = v
        else:
            self.__internal_data['Cache']['Tether'][entry] = value

    def get_tether_cache(self, entry=None):
        """Method for obtaining the information contained in the tether cache.
        :param entry: If None, a dict containing all cached information is retrieved. If a str, cache information about
        that parameter is returned."""
        if entry is None:
            return self.__internal_data['Cache']['Tether']
        else:
            return self.__internal_data['Cache']['Tether'][entry]

    def set(self, value):
        """Internal placeholder method."""
        pass

    def get(self):
        """Internal placeholder method."""
        pass

    def tooltip(self, text, *args, **kwargs):
        if 'Tooltip' in self.__internal_data:
            self.__internal_data['Tooltip'].tt_update(text, *args, **kwargs)
        else:
            self.__internal_data['Tooltip'] = Tooltip(self, text, *args, **kwargs)

    def get_fontsize(self):
        """Internal method that grabs the font size of widget. This only works if 'font' is a valid kwarg."""

        try:
            font = self['font']
        except KeyError:
            return None

        if isinstance(font, str):
            font_size = int(font.split(' ')[1])
        elif isinstance(font, (tuple, list)):
            font_size = int(font[1])
        else:
            return None

        return font_size

    def dv(self, dv=None, group='', branch=''):
        """Method that can access a globally defined variable."""
        if dv is None:  # post the entire dv dict
            return self.winfo_toplevel().dependent_variables
        elif dv is True:  # fetch all dv defined under the current group and branch
            return self.winfo_toplevel().dependent_variables[group][branch]
        else:  # else fetch the specific variable
            return self.winfo_toplevel().dependent_variables[group][branch][dv]

    def dv_define(self, dv, var, group='', branch=''):
        """Method that can define global variables."""
        if group not in self.dv(): self.winfo_toplevel().dependent_variables[group] = {};
        if branch not in self.dv()[group]: self.winfo_toplevel().dependent_variables[group][branch] = {};
        self.winfo_toplevel().dependent_variables[group][branch][dv] = var

    def dv_remove(self, dv=None, group=None, branch=None):
        """Method that can remove global variables."""
        _ = self.winfo_toplevel().dependent_variables
        if dv is not None:
            if group is not None:
                if branch is not None: del _[group][branch][dv];
                else: del _[group][dv];
            else: del _[dv];
        else:
            if group is not None:
                if branch is not None: del _[group][branch];
                else: del _[group];
            else: _ = {};
        self.winfo_toplevel().dependent_variables = _

    def dv_get(self, dv=None, group='', branch='', **kwargs):
        """Method that fetches values from global variables"""

        as_dict = kwargs['as_dict'] if 'as_dict' in kwargs else False

        if branch is not None:
            if dv is not None:
                try:
                    return self.winfo_toplevel().dependent_variables[group][branch][dv].get()
                except KeyError:  # the likely KeyError cause is that the branch should be None at the call and not ''
                    """Note that this functionality can be devious. For example, if the sought after dv has not yet been 
                    defined but surrounding parameters have, dv_get will return a list of results for all OTHER dv than 
                    the target dv."""
                    return self.dv_get(dv=dv, group=group, branch=None, **kwargs)
            else:
                if as_dict is False:
                    return [i.get() for i in self.winfo_toplevel().dependent_variables[group][branch].values()]
                else:
                    return {k: v.get() for k, v in self.winfo_toplevel().dependent_variables[group][branch].items()}
        else:
            if dv is not None:
                if as_dict is False:
                    _ = []
                    for branch in self.winfo_toplevel().dependent_variables[group].values():
                        try: _.append(branch[dv].get());
                        except KeyError: pass;
                else:
                    _ = {}
                    for branch, values in self.winfo_toplevel().dependent_variables[group].items():
                        try: _[branch] = values[dv].get();
                        except KeyError: pass;
                return _
            elif group is not None:
                # this functionality can only display as a dict
                _ = {}
                for branch, b in self.winfo_toplevel().dependent_variables[group].items():
                    _[branch] = {}
                    for item, i in b.items():
                        try: _[branch][item] = i.get();
                        except KeyError: pass;
                return _

    def dv_set(self, dv, value, group='', branch='', **kwargs):
        """Method that sets values for a global variable.
        :param dv: name of dependent variable to be set.
        :param value: value to be set.
        :param group: name of the dv group if any. The default is ''.
        :param branch: name of the dv branch, if any. The default is ''.
        :keyword update: if the dv is a DictVar this kwarg can be set to True, which will update the DictVar value,
        rather than replace it. The default behavior is to replace the DictVar value."""

        kwargs['update'] = kwargs['update'] if 'update' in kwargs else False

        if kwargs['update'] is False:
            self.winfo_toplevel().dependent_variables[group][branch][dv].set(value)
        else:
            self.winfo_toplevel().dependent_variables[group][branch][dv].update(value)

    def dv_trace(self, dv, *args, **kwargs):
        """Method that adds a trace to a global variable."""

        # construct tree
        if 'group' in kwargs:
            group = kwargs['group']
            del kwargs['group']
        else: group = '';

        if 'branch' in kwargs:
            branch = kwargs['branch']
            del kwargs['branch']
        else: branch = '';

        if group not in self.winfo_toplevel().traces: self.winfo_toplevel().traces[group] = {};
        if branch not in self.winfo_toplevel().traces[group]: self.winfo_toplevel().traces[group][branch] = {};
        if dv not in self.winfo_toplevel().traces[group][branch]: self.winfo_toplevel().traces[group][branch][
            dv] = CommandWrapper();

        if self.winfo_toplevel().traces[group][branch][dv].cbname is not None:
            self.dv(dv, group=group, branch=branch).trace_remove(self.winfo_toplevel().traces[group][branch][dv].mode,
                                                                 self.winfo_toplevel().traces[group][branch][dv].cbname)

        self.winfo_toplevel().traces[group][branch][dv] + args[1]
        self.__internal_data['Traces'].add(args[1])  # store callback name internally
        _ = self.dv(dv, group=group, branch=branch).trace_add(args[0],
                                                              self.winfo_toplevel().traces[group][branch][dv], **kwargs)
        self.winfo_toplevel().traces[group][branch][dv].cbname = _
        self.winfo_toplevel().traces[group][branch][dv].mode = args[0]

    def dv_remove_traces(self, dv, *callbacks, **kwargs):
        """Method that removes a trace from a global variable.
        :param dv: dependent variable.
        :param callbacks: callbacks to remove."""

        group = kwargs['group'] if 'group' in kwargs else ''
        branch = kwargs['branch'] if 'branch' in kwargs else ''

        try:
            if self.winfo_toplevel().traces[group][branch][dv].cbname is not None:
                for cb in callbacks:
                    self.winfo_toplevel().traces[group][branch][dv] - cb
                    self.__internal_data['Traces'].discard(cb)  # remove callback name internally

                _m = self.winfo_toplevel().traces[group][branch][dv].mode
                self.dv(dv, group=group, branch=branch).trace_remove(
                    _m, self.winfo_toplevel().traces[group][branch][dv].cbname)

                _ = self.dv(dv, group=group, branch=branch).trace_add(
                    _m, self.winfo_toplevel().traces[group][branch][dv])
                self.winfo_toplevel().traces[group][branch][dv].cbname = _
        except KeyError: pass;

    def remove_class_traces(self):
        """Method that removes all traces from a class.name."""
        for branches in self.winfo_toplevel().traces.values():
            for traces in branches.values():
                for k, v in traces.items():
                    cbs = [i for i in self.__internal_data['Traces'] if i.__qualname__ in v.callbacks.keys()]
                    self.dv_remove_traces(k, *cbs)

    def dv_check(self, dv=None, group=None, branch=None):
        """Method that checks whether a specific global variable has been defined."""
        if group is not None:
            if group in self.dv():
                if branch is not None:
                    if branch in self.dv()[group]:
                        if dv is not None:
                            if dv in self.dv()[group][branch]: return True;
                            else: return False;
                        else: return True;
                    else: return False;
                else:
                    if dv is not None:
                        if dv in self.dv()[group]: return True;
                        else: return False;
                    else: return True;
            else: return False;
        else:
            if dv in self.dv(): return True;
            else: return False;

    def post_self_click(self, e):
        """Internal method that updates the global last clicked variable on click."""
        if self.dv_check('LastClick'):
            self.dv_set('LastClick', self.tkID)

    @property
    def __name__(self):
        return type(self).__name__

    def _check_bind_id(self):
        if self.tkID not in self._all_binds:
            self._all_binds[self.tkID] = {}

    def bind_all(self, sequence=None, func=None, add=False, *args, **kwargs):
        """Replacement method for bind_all that allows for the rebind_all method to work."""

        self._check_bind_id()
        _ = self._all_binds[self.tkID]
        if sequence in _:
            if add is False:
                _[sequence] = [func]
                super().bind_all(sequence, func, add=False)
            else:
                if func not in _[sequence]:  # prevent duplicating function calls
                    _[sequence] += [func]
                super().unbind_all(sequence)
                for v in _[sequence]:
                    super().bind_all(sequence, v, add=True)
        else:
            _[sequence] = [func]
            super().bind_all(sequence, func)
        self._all_binds[self.tkID] = _

    def all_binds(self):
        return self._all_binds

    def unbind_all(self, sequence=None, clear=False):
        """Replacement method for unbind_all that implements unbinding of the rebind_all functionality."""

        self._check_bind_id()
        if sequence is not None:
            super().unbind_all(sequence)

            if clear is True:
                del self._all_binds[self.tkID][sequence]
        else:
            _ = self._all_binds[self.tkID].copy()
            for k in _:
                super().unbind_all(k)

                if clear is True:
                    del self._all_binds[self.tkID][k]

    def rebind_all(self, sequence=None):
        """Method that rebinds functions to sequences previously defined."""

        self._check_bind_id()
        _ = self._all_binds[self.tkID]
        if sequence is not None:
            if sequence in _:
                for v in _[sequence]:
                    self.bind_all(sequence, v, add=True)
        else:
            for k, v in _.items():
                for func in v:
                    self.bind_all(k, func, add=True)


class Tooltip(tk.Frame):
    """Tooltip for cellexum application frames."""

    def __init__(self, parent, text, *args, **kwargs):

        params = {  # define standard parameters
            'bg': supports.__cp__['dark_bg'],
            'fg': 'white',
            'font': 'Arial 10',
            'delay': parent.dv_get('TooltipTimer'),  # in ms
            'shift': (2, 2),  # x, y shift of the tooltip compared to cursor position
            'restrict_wrap': 30  # shift the label wrapping to be 50 pixels to the left of the edge
        }

        for k in params:  # set custom parameters
            if k in kwargs:
                params[k] = kwargs[k]
                del kwargs[k]

        super().__init__(parent.winfo_toplevel(), *args, **kwargs)

        self.__params = params

        self.__state = tk.BooleanVar(self, True)
        self.__hover_state = tk.BooleanVar(self)
        self.__has_left = False
        self.__cursor = None

        self.__label = tk.Label(self, text=text, bg=params['bg'], fg=params['fg'], font=params['font'], justify='left')
        self.__label.pack()

        self.__text = text  # store passed text

        self.bind('<Enter>', self.__tooltip_enter)
        self.bind('<Leave>', self.__tooltip_leave)

        parent.bind('<Enter>', self.__parent_enter, add=True)
        parent.bind('<Leave>', self.__parent_leave, add=True)

        self.__label.bind_all('<Key>', self.force_forget_tooltip, add=True)

    def tt_update(self, text):
        """Method to update the tooltip text."""
        self.__label.configure(text=text)
        self.__text = text

    def __tooltip_enter(self, e):
        self.__state.set(True)

    def __tooltip_leave(self, e):
        if self.winfo_ismapped():
            self.__state.set(False)

            # wait 2 ms to trigger tooltip removal to ensure parent frame is not hovered
            self.after(2, self.__forget_tooltip)

    def __parent_enter(self, e):
        # only trigger tooltip if it was not removed by the user in that cursor position
        if (self.winfo_pointerx(), self.winfo_pointery()) != self.__cursor:
            self.__hover_state.set(True)
            if not self.winfo_ismapped():
                self.after(self.__params['delay'], self.__place_tooltip)
            self.__has_left = False

    def __parent_leave(self, e):
        if self.winfo_ismapped():
            self.__hover_state.set(False)

            # wait 2 ms before tooltip removal to ensure tooltip is not being hovered
            self.after(2, self.__forget_tooltip)
        self.__has_left = True

    def __forget_tooltip(self):
        if self.__state.get() is False and self.__hover_state.get() is False:
            self.place_forget()

    def __place_tooltip(self):
        if self.__has_left is False:
            x = self.winfo_toplevel().winfo_pointerx() - self.winfo_toplevel().winfo_rootx() - self.__params['shift'][0]
            y = self.winfo_toplevel().winfo_pointery() - self.winfo_toplevel().winfo_rooty() - self.__params['shift'][1]

            self.__label['text'] = self.__text  # reset label
            max_width = self.winfo_toplevel().winfo_width() - x  # find max width for label
            max_width -= self.__params['restrict_wrap']  # adjust label wrapping
            max_chars = len(self.__text) * max_width // self.__label.winfo_reqwidth()  # set max line size

            last_line = self.__label['text'].split('\n')[-1]
            while len(last_line) > max_chars:  # create new lines while the last line is still too long
                spaces = [i for i, e in enumerate(last_line) if e == ' ']  # grab space positions for  line breaks
                lower_spaces = [i for i in spaces if i < max_chars]  # find spaces to the left of the index
                _id = lower_spaces[-1]  # set nearest spacing for line break

                # create line break at position and update index for the next set of characters
                self.__label['text'] = ('\n'.join(self.__label['text'].split('\n')[:-1]) + '\n' + last_line[:_id] +
                                        '\n' + last_line[_id + 1:])
                last_line = self.__label['text'].split('\n')[-1]  # find the new last line

            if self.__label['text'][0] == '\n':  # remove the first line break
                self.__label['text'] = self.__label['text'][1:]

            self.place(x=x, y=y)  # place the tooltip

    def force_forget_tooltip(self, e):
        self.__tooltip_leave(None)
        self.__parent_leave(None)
        self.__cursor = (self.winfo_pointerx(), self.winfo_pointery())


class AppCanvas(AppWidget, tk.Canvas):
    """Custom canvas widget for the GUI."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)


class AppFrame(AppWidget, tk.Frame):
    """Custom frame widget for the GUI. Adds additional functionality to the Frame widget."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.scrollbar = None
        self.dimensions = (self.winfo_width(), self.winfo_height())
        self.parent = parent

        self.bind('<Button-1>', self.__set_focus)

    @staticmethod
    def __set_focus(e):
        """Internal method that allows focus to be set arbitrarily to frames."""
        e.widget.focus_set()


class ContentFrame(AppFrame):
    """Internal class that handles content frame creation with scrollbar."""

    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)

        self._memory = {}  # define a memory dict for restoring scroll positions
        self._scroll_status = False

        # Construct canvas and scrollbar in parent frame.
        self.canvas = canvas = AppCanvas(self, bd=0, highlightthickness=0, )
        canvas.pack(side='left', fill='both', expand=True)
        canvas.xview_moveto(0); canvas.yview_moveto(0)  # set default view

        # Construct scrollable frame inside the canvas.
        self.interior = interior = tk.Frame(canvas)
        canvas.create_window(0, 0, window=interior, anchor='nw')

        # construct vertical scrollbar
        self.vscrollbar = vscrollbar = tk.Scrollbar(canvas, orient='vertical', command=canvas.yview)
        vscrollbar.pack(fill='y', side='right', expand=True)
        canvas.config(yscrollcommand=vscrollbar.set)
        canvas.create_window(0, 0, window=vscrollbar, anchor='ne')  # place scrollbar in canvas

        # Bind events to functions.
        canvas.bind('<Configure>', self.__configure_interior)
        canvas.bind('<Enter>', self.__bind_mouse)
        canvas.bind('<Leave>', self.__unbind_mouse)

        # set interior size as cache
        self.__current_interior_size = (self.interior.winfo_width(), self.interior.winfo_height())

    def __configure_interior(self, e):
        """Internal method that updates the scrollable area."""
        size = (self.interior.winfo_width(), self.interior.winfo_height())  # grab interior size
        self.canvas.config(scrollregion=(0, 0, size[0], size[1]))  # set scrollable region
        if self.interior.winfo_width() != self.canvas.winfo_width():  # update canvas width to interior width
            self.canvas.config(width=self.interior.winfo_width())

        # place or forget scrollbar, depending on the height of the widget compared to the main frame.
        if self.interior.winfo_height() <= self.canvas.winfo_height():
            self.vscrollbar.pack_forget()
        else:
            self.vscrollbar.pack(fill='y', side='right', expand=True)
        self.canvas.yview_scroll(-1 * int(e.delta / 120), 'units')

    def __bind_mouse(self, e):
        if self.vscrollbar.winfo_ismapped():  # if vertical scrollbar exists, bind mouse wheel to scrolling.
            self._scroll_status = True
            self.canvas.bind_all("<MouseWheel>", self.__scroll_mouse)

    def __unbind_mouse(self, e):
        self.canvas.unbind_all("<MouseWheel>")
        self._scroll_status = False

    def __scroll_mouse(self, e):
        try:  # a toplevel widget may still occupy focus when closed, this fixes this error report
            self.canvas.yview_scroll(-1 * int(e.delta / 120), 'units')
            if self.dv_check('ActiveSelectionMenu', '', '') is True:  # clear active selection menu if any
                if self.dv_get('ActiveSelectionMenu') is True:
                    self.dv_set('ActiveSelectionMenu', False)
        except tk.TclError:
            pass

    def refresh_content_frame(self):
        size = (self.interior.winfo_width(), self.interior.winfo_height())  # grab interior size

        # if the interior size has changed, update the scrollable region and the canvas width
        if size != self.__current_interior_size:
            self.canvas.config(scrollregion=(0, 0, size[0], size[1]))  # set scrollable region
            self.canvas.config(width=self.interior.winfo_width())
        if self.interior.winfo_height() <= self.canvas.winfo_height():
            self.vscrollbar.pack_forget()
        else:
            try:  # ideally avoid running code if error would occur instead of just catching it
                self.vscrollbar.pack(fill='y', side='right', expand=False)
                if self._scroll_status is False:
                    self._scroll_status = True
                    self.canvas.bind_all("<MouseWheel>", self.__scroll_mouse)
            except tk.TclError:
                pass

        self.__current_interior_size = size

    def set_position_memory(self, name):
        self._memory[name] = (self.canvas.xview(), self.canvas.yview())

    def get_position_memory(self, name, scroll_region):
        self.canvas.config(scrollregion=scroll_region)  # set scrollable region
        if name in self._memory:
            xview, yview = self._memory[name]
            xview = xview[0]; yview = yview[0]
        else:
            xview = 0; yview = 0
        self.canvas.xview_moveto(xview); self.canvas.yview_moveto(yview)


class AppButton(AppWidget, tk.Button):
    """Custom button class for the Cellexum application"""

    def __init__(self, parent, size=None, *args, **kwargs):
        pars = {
            'padx': 15,
            'pady': 3,
            'bd': 0,
            'bg': supports.__cp__['dark_bg'],
            'fg': supports.__cp__['fg'],
            'activebackground': supports.highlight(supports.__cp__['dark_bg'], -20),
            'activeforeground': supports.__cp__['fg'],
            'font': supports.button_font,
            'anchor': 'w',
            'multithreading': False,
            'command': None
        }

        # set size preset
        if size == 'large':
            pars['width'] = 13
        elif size == 'medium':
            pars['width'] = 15
        elif size == 'small':
            pars['width'] = 7
            pars['anchor'] = 'center'
        elif isinstance(size, int):
            pars['width'] = size
        elif not size:
            pars['anchor'] = 'center'
        else:
            raise ValueError(f'Invalid size "{size}". Valid options are "large", "medium", "small", or an int')

        for k in pars:
            if k not in kwargs:
                kwargs[k] = pars[k]

        self.__command = kwargs['command']

        if kwargs['multithreading'] is True:
            kwargs['command'] = self.__multithreader
        del kwargs['multithreading']

        super().__init__(parent, *args, **kwargs)

        self.__bg = kwargs['bg']
        self.bind('<Enter>', self.__on_enter)
        self.bind('<Leave>', self.__on_leave)

    def __on_enter(self, e):
        self['bg'] = supports.highlight(self.__bg, 20)
        self['cursor'] = 'hand2'

    def __on_leave(self, e):
        self['bg'] = self.__bg

    def __multithreader(self):
        return threading.Thread(target=self.__command).start()

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = (30, 5);
        super().grid(*args, **kwargs)


class MenuButton(AppButton):
    """Custom class for menu buttons for content switching."""

    def __init__(self, parent, frame, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.frame = frame

        self.configure(command=self.update_active_frame)  # bind active frame update to command
        self.bind('<Control-1>', self.reload_frame)  # bind Ctrl-Click to frame refresh

    def update_active_frame(self):
        self.dv_set('ActiveFrame', self.frame.tkID)
        self.frame.update_idletasks()

    def reload_frame(self, *_):
        self.frame.reload()
        self.frame.update_idletasks()


class AppEntry(AppWidget, tk.Entry):
    """Custom entry class for the Cellexum application"""

    def __init__(self, parent, *args, **kwargs):

        pars = {
            'fg': supports.__cp__['fg'],
            'bg': supports.__cp__['dark_bg'],
            'relief': 'flat',
            'bd': 2,
            'vartype': str,
            'textvariable': tk.StringVar(),
            'selectbackground': '#ffffff',
            'selectforeground': '#000000',
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        self.disabled_selection = None
        if kwargs['vartype'] == int:
            self.disabled_selection = 0
        elif kwargs['vartype'] == float:
            self.disabled_selection = 0

        self.entry = kwargs['textvariable']

        if 'default' in kwargs:  # set default entry if prompted
            self.set(kwargs['default'])
            del kwargs['default']
        self.default = self.entry.get()  # store the default value

        self.tether_action = {
            'state': 'disabled',
            'selection': self.disabled_selection,
            'disabledbackground': supports.__cp__['disabled_bg'],
            'disabledforeground': supports.__cp__['disabled_fg']
        }

        self.__vartype = kwargs['vartype']
        del kwargs['vartype']  # remove vartype from kwargs before parent call

        super().__init__(parent, *args, **kwargs)

        self.bind('<Enter>', self.__on_enter)

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = (0, 5);
        super().grid(*args, **kwargs)

    def get(self):
        value = self.entry.get()
        if self.__vartype is bool:
            if value == 'True':
                return True
            elif value == 'False':
                return False
            else:
                return bool(value)
        else:
            try:
                return self.__vartype(value)
            except ValueError:
                return value

    def set(self, value):
        self.entry.set(value)

    def trace_add(self, *args, **kwargs):
        self.entry.trace_add(*args, **kwargs)

    def __on_enter(self, e):
        if self['state'] in (tk.DISABLED, 'disabled'):
            self['cursor'] = 'arrow'
        else:
            self['cursor'] = 'xterm'


class SettingEntry(AppEntry):
    """Entry class for typing settings."""

    def __init__(self, parent, *args, **kwargs):
        pars = {
            'bg': supports.highlight(supports.__cp__['dark_bg'], -30),
            'width': 5,
            'activebackground': supports.__cp__['dark_bg'],
            'font': 'Arial 12',
        }

        for k in pars:
            if k not in kwargs:
                kwargs[k] = pars[k]
        self.kwargs = kwargs.copy()

        del kwargs['activebackground']

        super().__init__(parent, *args, **kwargs)

        self.trace_add('write', self.__on_active)
        self.__on_active()  # check state of entry

    def __on_active(self, *_):
        if self.get():
            self['bg'] = self.kwargs['activebackground']
        else:
            self['bg'] = self.kwargs['bg']


class TextButton(AppWidget, tk.Button):
    """Custom button class for the Cellexum application. The class supports warnings. These can only be triggered if
    there is already a command in the call."""

    def __init__(self, parent, *args, **kwargs):
        self.data = None  # allow for data-setting for unique functionality

        pars = {
            'fg': supports.__cp__['fg'],
            'font': 'Arial 6 bold',
            'bg': supports.__cp__['bg'],
            'bd': 0,
            'activebackground': supports.__cp__['bg'],
            'activeforeground': supports.__cp__['dark_bg'],
            'cursor': 'hand2',
            'no_upper': False,
        }

        # set data attribute if it exists in kwargs
        if 'data' in kwargs.keys():
            self.data = kwargs['data']
            del kwargs['data']

        # set special command functionality. Note that this overwrites any 'command' attribute in the call
        if 'function' in kwargs:
            if kwargs['function'] == 'open_image':
                def wrapper():
                    os.startfile(self.data)

                pars['command'] = wrapper
                if 'command' in kwargs:
                    del kwargs['command']
                del kwargs['function']

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        if 'text' in kwargs:  # convert chars to upper case if text is passed
            if kwargs['no_upper'] is False:
                kwargs['text'] = kwargs['text'].upper()
        del kwargs['no_upper']

        super().__init__(parent, *args, **kwargs)

        self.__fg = kwargs['fg']

        self.bind('<Enter>', self.__on_enter)
        self.bind('<Leave>', self.__on_leave)

    def __on_enter(self, e):
        self['fg'] = supports.highlight(self.__fg, -20)

    def __on_leave(self, e):
        self['fg'] = self.__fg

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = (2, 5);
        super().grid(*args, **kwargs)

    def bold(self):
        """Method that updates the font to bold."""

        font, restore = self._get_font()
        if font[-1] != 'bold': font += ['bold'];  # add bold to the font

        if restore:
            self['font'] = ' '.join(font)  # restore font formatting
        else:
            self['font'] = font

    def normal(self):
        """Method that updates the font to normal."""

        font, restore = self._get_font()
        if font[-1] == 'bold': font = font[:-1];  # add bold to the font

        if restore:
            self['font'] = ' '.join(font)  # restore font formatting
        else:
            self['font'] = font

    def _get_font(self, res_param=True):
        font = self['font']  # fetch current font
        restore = False
        if isinstance(font, str):
            font = font.split(' ')  # split font
            restore = True

        if res_param: return font, restore;
        else: return font;


class AppLabel(AppWidget, tk.Label):
    """Custom Label class for the Cellexum application"""

    def __init__(self, parent, *args, **kwargs):
        pars = {
            'bd': 0,
            'bg': supports.__cp__['bg'],
            'fg': supports.__cp__['fg'],
            'anchor': 'nw',
            'font': 'Arial 12',
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        super().__init__(parent, *args, **kwargs)

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = (2, 5);
        super().grid(*args, **kwargs)


class ImageButton(AppLabel):
    """Custom button class for the Cellexum application"""

    def __init__(self, parent, image, command=None, *args, **kwargs):

        pars = {
            'bg': supports.__cp__['dark_bg'],
            'fg': supports.__cp__['fg'],
            'hover': supports.__cp__['bg'],
            'active': supports.__cp__['dark_fg'],
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        img = cv2.imread(rf'{supports.__gpx__}\{image}.png', cv2.IMREAD_UNCHANGED)

        self._images = {
            'normal': ImageTk.PhotoImage(Image.fromarray(supports.colorize(img, kwargs['fg']))),
            'hover': ImageTk.PhotoImage(Image.fromarray(supports.colorize(img, kwargs['hover']))),
            'active': ImageTk.PhotoImage(Image.fromarray(supports.colorize(img, kwargs['active']))),
        }
        self.command = command

        for k in ('hover', 'active'):
            del kwargs[k]

        super().__init__(parent, image=self._images['normal'], *args, **kwargs)

        self.bind('<Enter>', self._hover)
        self.bind('<Leave>', self._normal)
        self.bind('<Button-1>', self._active)
        self.bind('<ButtonRelease-1>', self._hover)

    def _hover(self, e):
        self.configure(image=self._images['hover'])
        self['cursor'] = 'hand2'

    def _active(self, e):
        self.configure(image=self._images['active'])
        if self.command is not None:
            self.command()

    def _normal(self, e):
        self.configure(image=self._images['normal'])


class ImageMenuButton(ImageButton):
    """Custom button class for the Cellexum application"""

    def __init__(self, parent, image, frame, *args, **kwargs):
        pars = {
            'command': None
        }
        kwargs.update(pars)

        self.frame = frame

        ImageButton.__init__(self, parent, image, *args, **kwargs)

        self.command = self.update_active_frame  # bind active frame update to command
        self.bind('<Control-1>', self.reload_frame)  # bind Ctrl-Click to frame refresh

    def update_active_frame(self):
        self.dv_set('ActiveFrame', self.frame.tkID)

    def reload_frame(self, *_):
        self.frame.reload()


class AppTitle(AppLabel):
    """Custom Title class for the Cellexum application"""

    def __init__(self, parent, *args, **kwargs):
        kwargs['font'] = 'Arial 24 bold'
        super().__init__(parent, *args, **kwargs)

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = 10;
        super().grid(*args, **kwargs)


class AppSubtitle(AppLabel):
    """Custom Subtitle class for the Cellexum application"""

    def __init__(self, parent, *args, **kwargs):
        kwargs['font'] = 'Arial 18'
        super().__init__(parent, *args, **kwargs)

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = (30, 5);
        super().grid(*args, **kwargs)


class AppHeading(AppLabel):
    """Custom Heading class for the Cellexum application"""

    def __init__(self, parent, *args, **kwargs):
        kwargs['font'] = 'Arial 12 bold'
        super().__init__(parent, *args, **kwargs)

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = 10;
        super().grid(*args, **kwargs)


class _SelectionMenuItem(tk.Frame, AppWidget):
    def __init__(self, parent, *args, **kwargs):
        self.command = None
        super().__init__(parent.winfo_toplevel(), *args)

        self.parent = parent

        # solve kwargs before passing them to the label
        self.__bg = kwargs['bg']
        self.__highlight = kwargs['highlight']
        del kwargs['highlight']

        kwargs['font'] = 'Arial 12'
        self.label = label = tk.Label(self, **kwargs)
        label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        label.bind('<Button-1>', self.__update_selection)
        label.bind('<Enter>', self.__mouse_enter)
        label.bind('<Leave>', self.__mouse_leave)

    def __update_selection(self, e):
        """Internal method that updates the parent selection menu selection."""
        self.parent.set(self.label['text'])
        if self.command is not None:
            self.command()
        self.parent.active.set(False)  # update parent status
        self.dv_set('LastClick', self.parent.tkID)  # set the last click to be parent ID

    def __mouse_enter(self, e):
        self.label['bg'] = supports.highlight(self.__bg, self.__highlight)
        self.label['cursor'] = 'hand2'

    def __mouse_leave(self, e):
        self.label['bg'] = self.__bg


class SelectionMenu(AppFrame):
    """Custom selection menu for the cellexum application."""

    def __init__(self, parent, options, default=None, commands=None, *args, **kwargs):
        self.__options = None
        self.__option_names = options
        self._commands = commands
        if commands is None:
            commands = {}
        if 'highlight' in kwargs:
            self.__highlight = kwargs['highlight']
            del kwargs['highlight']
        else:
            self.__highlight = 30

        if 'bg' not in kwargs:
            if 'background' in kwargs:
                kwargs['bg'] = kwargs['background']
                del kwargs['background']
            else:
                kwargs['bg'] = supports.__cp__['dark_bg']

        if 'placeholder' in kwargs:
            self.placeholder = kwargs['placeholder']
            del kwargs['placeholder']
        else:
            self.placeholder = 'Select Option'

        pars = {
            'fg': supports.__cp__['fg'],
            'justify': 'center',
            'font': 'Arial 12',
        }

        for k in pars:
            if k not in kwargs:
                kwargs[k] = pars[k]

        super().__init__(parent, *args)

        # construct selection label
        self.previous = None
        self.current = None
        self.active = tk.BooleanVar(self, False)
        self.__bg = kwargs['bg']
        self.__state = 'normal'  # add a state variable to trigger from tether
        self.selection = tk.StringVar(self)
        self.__hidden_options = list(options)
        if default is None:
            self.selection.set(self.placeholder)
            self.__hidden_options.append(self.placeholder)
        else:
            self.selection.set(options[default])

        letter_width = max([len(i) for i in self.__hidden_options])
        if not 'width' in kwargs:
            kwargs['width'] = letter_width

        self.__label = label = tk.Label(self, textvariable=self.selection, **kwargs)
        label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.__kwargs = kwargs
        self.__parent = parent
        self.default = self.selection.get()  # store the default value

        # set up option labels
        self.__set_up_options(options, commands)

        label.bind('<Button-1>', self.toggle)
        label.bind('<Enter>', self.__mouse_enter)
        label.bind('<Leave>', self.__mouse_leave)
        self.active.trace_add('write', self.__update_placement)
        self.trace_add('write', self.__update_self)

        self.tether_action = {'state': 'disabled',
                              'fg': supports.__cp__['disabled_fg'],
                              'bg': supports.__cp__['disabled_bg']}

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = (0, 5);
        super().grid(*args, **kwargs)

    def __setitem__(self, key, value):
        if key == 'state':
            self.__state = value
        else:
            self.__label[key] = value
            for elem in self.__options:
                elem.label[key] = value

    def disable(self, value=None):
        self['state'] = 'disabled'
        self['fg'] = supports.__cp__['disabled_fg']
        self['bg'] = supports.__cp__['disabled_bg']
        self['cursor'] = 'arrow'
        if value is not None:
            self.set(value)

    def enable(self, stall=False):
        self['state'] = 'normal'
        self['cursor'] = 'hand2'
        self['fg'] = self.__kwargs['fg']
        self['bg'] = self.__bg
        if not stall:
            if self.current != self.previous:
                self.set(self.previous)

    def label_bind(self, *args, **kwargs):
        return self.__label.bind(*args, **kwargs)

    def __getitem__(self, item):
        if item == 'state':
            return self.__state
        else:
            return self.__label[item]

    def set(self, value):
        self.selection.set(value)

    def get(self):
        return self.selection.get()

    def trace_add(self, *args, **kwargs):
        self.selection.trace_add(*args, **kwargs)

    def __update_self(self, *_):
        """Internal method that stores the current and previous selection."""
        if self.current is None:
            self.current = self.get()
            self.previous = self.get()
        else:
            self.previous = self.current
            self.current = self.get()

    def set_previous(self):
        """Method that sets the previous selection as the current selection."""
        self.set(self.previous)

    def __mouse_enter(self, e):
        if self['state'] == 'normal':
            self.__label['bg'] = supports.highlight(self.__bg, self.__highlight)
            self['cursor'] = 'hand2'

    def __mouse_leave(self, e):
        if self['state'] == 'normal':
            self.__label['bg'] = self.__bg

    def forget_options(self, e):
        for option in self.__options:
            option.place_forget()
        try: self.lower();
        except tk.TclError: pass;

    def place_options(self):
        if self['state'] == 'normal':
            self.tkraise()
            x = self.winfo_rootx() - self.winfo_toplevel().winfo_rootx()
            y = self.winfo_rooty() - self.winfo_toplevel().winfo_rooty()
            shift = self.winfo_reqheight()
            for option in self.__options:
                option.place(x=x, y=y + shift)
                shift += option.winfo_reqheight()

    def __update_placement(self, *_):
        if self.active.get():
            self.place_options()
            self.dv_define('ActiveSelectionMenu', self.active)
        else:
            self.forget_options(None)

    def toggle(self, e):
        if self['state'] == 'normal':
            if self.active.get():
                self.active.set(False)
            else:
                # delay the activation setting to avoid getting cleared by the global variable
                self.after(5, self.active.set, True)
        else:  # never toggle on the menu if it is disabled
            self.active.set(False)

    def add_option(self, option, command=None, order=None):
        """Method that adds an option after initial object creation.
        :params option: the name of the added option.
        :params command: a command for execution if the option is pressed. Default is None.
        :params order: if the new option should be placed at a specific point in the constructed selection menu, the
        order can be set with this parameter as an index. Ex an order of -1 will place the new option between the
        second last and the last existing option. The default is None."""
        kwargs = self.__kwargs
        kwargs['bg'] = supports.highlight(self.__bg, -20)
        elem = _SelectionMenuItem(self, text=option, highlight=self.__highlight, **kwargs)
        if command is not None:
            elem.command = command

        if order is None:
            self.__options.append(elem)
        else:
            self.__options = self.__options[:order] + [elem] + self.__options[order:]

    def __set_up_options(self, options, commands):
        """Internal method that sets up the options in the selection menu."""
        kwargs = self.__kwargs
        self.__options = []
        kwargs['bg'] = supports.highlight(self.__bg, -20)
        for n, option in enumerate(options):
            elem = _SelectionMenuItem(self, text=option, highlight=self.__highlight, **kwargs)
            # permit command allocation both by index and by name
            if n in commands:
                elem.command = commands[n]
            if option in commands:
                elem.command = commands[option]
            self.__options.append(elem)

    def update_options(self, options, commands=None, default=None, force=False):
        """Method that updates all options in the selection menu, according to the provided options.
        :params options: the list of options to update.
        :params commands: dict of commands. If set to None, no option commands will be set. If set to True, reuse the
        commands from the initial call."""

        self.__hidden_options = list(options)
        if default is None: self.__hidden_options.append(self.placeholder);

        try:
            letter_width = max([len(i) for i in self.__hidden_options])
            self['width'] = letter_width;
            self.__kwargs['width'] = letter_width
        except TypeError: pass;

        if self.__option_names != options or force:
            if commands is None:
                commands = {}
            elif commands is True:
                commands = self._commands

            for e in self.__options:
                e.destroy()

            self.__set_up_options(options, commands)
            if default is None:
                self.set(self.placeholder)
            else:
                self.set(default)
            self.__option_names = options


class SelectionEntry(SettingEntry):
    """Internal class that sets up a setting entry with a dynamic selection menu."""

    def __init__(self, parent, options, default=None, display_number=10, *args, **kwargs):
        self.__options = None
        self.__kwargs = kwargs
        self.__parent = parent
        self.previous = options
        self.default = default
        self.display_number = display_number
        self.suppress_update = False

        if 'highlight' in kwargs:
            self.__highlight = kwargs['highlight']
            del kwargs['highlight']
        else:
            self.__highlight = 30

        if 'bg' not in kwargs:
            if 'background' in kwargs:
                kwargs['bg'] = kwargs['background']
                del kwargs['background']
            else:
                kwargs['bg'] = supports.__cp__['dark_bg']

        if 'width' not in kwargs:
            if options:
                kwargs['width'] = max([len(i) for i in options]) + 1

        kwargs['justify'] = 'center'

        super().__init__(parent, *args, **kwargs)

        if default is not None:
            self.entry.set(default)

        self.active = tk.BooleanVar(self, False)
        self.__state = 'normal'  # add a state variable to trigger from tether
        self.__bg = kwargs['bg']

        self.__set_up_options(options)

        self.bind('<Button-1>', self.open_selection)
        self.bind('<Escape>', self.close_selection)
        self.bind_all('<MouseWheel>', self.close_selection)
        self.bind('<Enter>', self.__mouse_enter)
        self.bind('<Leave>', self.__mouse_leave)
        self.active.trace_add('write', self.__update_placement)
        self.trace_add('write', self.update_options)

        self.tether_action = {'state': 'disabled',
                              'fg': supports.__cp__['disabled_fg'],
                              'bg': supports.__cp__['disabled_bg']}

    def __mouse_enter(self, e):
        if self['state'] == 'normal':
            self['bg'] = supports.highlight(self.__bg, self.__highlight)
            self['cursor'] = 'xterm'

    def __mouse_leave(self, e):
        if self['state'] == 'normal':
            self['bg'] = self.__bg

    def forget_options(self, e):
        for option in self.__options:
            option.place_forget()
        self.lower()

    def __update_placement(self, *_):
        if not self.suppress_update:
            if self.active.get():
                self.place_options()
                self.dv_define('ActiveSelectionMenu', self.active)
            else:
                self.forget_options(None)

    def update_options(self, *_):
        if not self.suppress_update:
            self.forget_options(None)
            if self.active.get():
                self.place_options()
            else:
                self.after(5, self.active.set, True)

    def __set_up_options(self, options):
        """Internal method that sets up the options in the selection menu."""
        kwargs = self.__kwargs
        self.__options = []
        for option in options:
            kwargs['bg'] = supports.highlight(self.__bg, -20)
            kwargs['fg'] = self['fg'] if 'fg' not in kwargs else kwargs['fg']
            elem = _SelectionMenuItem(self, text=option, highlight=self.__highlight, **kwargs)
            self.__options.append(elem)

    def open_selection(self, e):
        """Internal method that pops open the selection menu part of the entry."""
        if self['state'] == 'normal':
            if not self.active.get():
                # delay the activation setting to avoid getting cleared by the global variable
                self.after(5, self.active.set, True)
        else:  # never toggle on the menu if it is disabled
            self.active.set(False)

    def close_selection(self, e):
        """Internal method that closes the selection menu part of the entry."""
        if self.active.get():
            self.active.set(False)

    def place_options(self):
        if self['state'] == 'normal':
            self.tkraise()
            x = self.winfo_rootx() - self.winfo_toplevel().winfo_rootx()
            y = self.winfo_rooty() - self.winfo_toplevel().winfo_rooty()
            shift = self.winfo_reqheight()

            # filter available selections based on current entry typing
            current_entry = self.get()
            display_options = []; _options = []
            for option in self.__options:
                if option.label['text'].startswith(current_entry):
                    display_options.append(option)
                else:
                    _options.append(option)

            for option in _options:
                if current_entry in option.label['text']:
                    display_options.append(option)

            for option in display_options[:self.display_number]:
                option.place(x=x, y=y + shift)
                shift += option.winfo_reqheight()

    def replace_options(self, options, default=None):
        """Method that updates all options in the selection menu, according to the provided options.
        :params options: the list of options to update.
        :params commands: dict of commands ."""

        if options != self.previous:  # prevent unnecessary updates
            self.previous = options
            for e in self.__options:
                e.destroy()

            self['width'] = max([len(i) for i in options]) + 1; self.__kwargs['width'] = self['width']
            self.__set_up_options(options)
            if default is None:
                self.set('')
            else:
                self.set(default)


class ZoomImageFrame(AppCanvas):
    """Internal class that will display an image in a canvas and allow for zoom"""

    def __init__(self, parent, *args, **kwargs):
        self.img = None
        self.__canvas_img = None
        self.__tk_zoom_img = None
        self.__tk_img = None
        self.__isset = False
        self.scroll_scalar = (.75, .75)
        self.mtime = None

        pars = {
            'width': 300,
            'height': 300,
            'highlightthickness': 0,
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        self._reference_dimensions = (kwargs['width'], kwargs['height'])

        super().__init__(parent, *args, **kwargs)

        # bind mouse controls to functions
        self.bind("<ButtonPress-1>", self.__set_zoom_image)
        self.bind("<B1-Motion>", self.__scroll_move)
        self.bind("<ButtonRelease-1>", self.__reset_image)

        # grab origin coordinates before moving the frame
        self.__init_x = self.xview()[0]
        self.__init_y = self.yview()[0]

    def __scroll_move(self, e):
        """Internal method that moves the canvas according to the mouse movement"""
        if self.__isset is True:  # avoid invalid interactions
            self.xview_moveto(e.x * self.scroll_scalar[0] / self.__tk_img.width())
            self.yview_moveto(e.y * self.scroll_scalar[1] / self.__tk_img.height())

    def __set_zoom_image(self, e):
        """Internal method that configures the zoom image and its initial view position"""
        if self.__isset is True:  # avoid invalid interactions
            self.__scroll_move(e)
            self.itemconfig(self.__canvas_img, image=self.__tk_zoom_img)

    def __reset_image(self, e):
        """Internal method that resets the image view and image"""
        if self.__isset is True:  # avoid invalid interactions
            self.itemconfig(self.__canvas_img, image=self.__tk_img)
            self.xview_moveto(self.__init_x)
            self.yview_moveto(self.__init_y)

    def set_image(self, path, **kwargs):
        """Update set functionality to instead set the image within the canvas."""

        self.mtime = os.path.getmtime(path)
        self.img = Image.open(path)  # open image
        if 'brighten' in kwargs:
            self.img = self.img.point(lambda p: p * kwargs['brighten'])

        if 'rotate' in kwargs:
            self.img = self.img.rotate(kwargs['rotate'])

        # re-scale the canvas to properly fit the image aspect ratio
        if self.img.height > self.img.width:
            self['width'] = self._reference_dimensions[0] * self.img.width / self.img.height
        else:
            self['height'] = self._reference_dimensions[1] * self.img.height / self.img.width

        # set up tkinter images
        rz_img = ImageOps.contain(self.img, (int(self['width']), int(self['height'])))  # resize image
        self.__tk_img = ImageTk.PhotoImage(rz_img, master=self)  # set tk display image
        self.__tk_zoom_img = ImageTk.PhotoImage(self.img, master=self)  # set zoom image

        # add image to canvas
        if self.__isset is False:
            self.__canvas_img = self.create_image(0, 0, anchor='nw', image=self.__tk_img)
            self.__isset = True
        else:
            self.itemconfig(self.__canvas_img, image=self.__tk_img)

        # setup zoom image region
        self.configure(scrollregion=(0, 0, self.img.width, self.img.height))


class AppCheckbutton(AppFrame):
    """Custom checkbutton for the cellexum application."""

    def __init__(self, parent, text='', default=True, *args, **kwargs):
        pars = {
            'bg': supports.__cp__['bg'],
            'fg': supports.__cp__['fg'],
            'disabledbackground': supports.__cp__['disabled_bg'],
            'disabledforeground': supports.__cp__['disabled_fg'],
            'boxcolor': supports.__cp__['dark_bg'],
            'checkcolor': supports.__cp__['fg'],
            'state': 'normal'
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        selection = None
        if 'selection' in kwargs:
            selection = kwargs['selection']
            del kwargs['selection']

        super().__init__(parent, *args, bg=kwargs['bg'])

        self.previous = default  # allow for storing a previous value
        self.text = text
        self.selection = selection if selection is not None else tk.BooleanVar(self, default)
        self.__kwargs = kwargs  # store kwargs in a variable
        self.state = tk.StringVar(self, kwargs['state'])  # set the default activity state
        self.state.trace_add('write', self.__state_check)
        self.tether_action = {
            'state': 'disabled',
            'selection': True
        }

        self.__label = AppLabel(self, text=text, tooltip=None, fg=kwargs['fg'])
        self.__box = AppFrame(self, width=12, height=12, bg=kwargs['boxcolor'], cursor='hand2')
        self.__check = AppFrame(self.__box, width=6, height=6, bg=kwargs['checkcolor'])

        self.__box.grid(row=0, column=0, sticky=tk.W)
        self.__check.grid(row=0, column=1, padx=3, pady=3, sticky=tk.W)
        self.__label.grid(row=0, column=1, padx=10, sticky=tk.W, pady=0)

        self.trace_add('write', self.__update_check)
        self.__update_check()
        self.__state_check()

    def grid(self, *args, **kwargs):
        if 'pady' not in kwargs: kwargs['pady'] = (0, 5);
        super().grid(*args, **kwargs)

    def get(self):
        return self.selection.get()

    def set(self, value):
        self.previous = self.get()  # store previous value
        self.selection.set(value)

    def set_previous(self):
        self.set(self.previous)

    def __toggle(self, e):
        if self.get() is True:  # set state to off if on
            self.set(False)
        else:  # set state to on if off
            self.set(True)

    def trace_add(self, *args, **kwargs):
        self.selection.trace_add(*args, **kwargs)

    def __update_check(self, *_):
        if self.get() is True:
            self.__check.grid()
        else:
            self.__check.grid_remove()

    def tooltip(self, text, *args, **kwargs):
        """Alter tooltip functionality to only be active on the label."""
        self.__label.tooltip(text, *args, **kwargs)

    def __setitem__(self, key, value):
        """Adjust the __setitem__ method to work as intended for the object."""
        self.__kwargs[key] = value  # update kwargs dict
        if key == 'boxcolor':
            self.__box['bg'] = value
        elif key == 'state':
            if value in (tk.NORMAL, 'normal'):
                self.state.set('normal')
            elif value in (tk.DISABLED, 'disabled'):
                self.state.set('disabled')
            else:
                raise ValueError(f'Invalid state {value!r}.')
            self.__kwargs[key] = self.state.get()  # update state key to match updated value
        elif key == 'checkcolor':
            self.__check['bg'] = value
        elif key in ('fg', 'text'):
            self.__label[key] = value

    def __getitem__(self, key):
        """Adjust the __getitem__ method to yield kwargs setup."""
        return self.__kwargs[key]

    def __state_check(self, *_):
        """Internal method that updates the state of the checkbutton."""
        if self['state'] == 'normal':
            self.__box.bind('<Button-1>', self.__toggle)
            self.__check.bind('<Button-1>', self.__toggle)
            self.__box['bg'] = self['boxcolor']
            self.__check['bg'] = self['checkcolor']
            self.__label['fg'] = self['fg']
            self.__box['cursor'] = 'hand2'
            self.__check['cursor'] = 'hand2'
        else:
            self.__box.unbind('<Button-1>')
            self.__check.unbind('<Button-1>')
            self.__box['bg'] = self['disabledbackground']
            self.__check['bg'] = self['disabledforeground']
            self.__label['fg'] = self['disabledforeground']
            self.__box['cursor'] = 'arrow'
            self.__check['cursor'] = 'arrow'


class TextCheckbutton(AppLabel):
    def __init__(self, parent, *args, **kwargs):

        default = True  # set default check state
        if 'default' in kwargs:
            default = kwargs['default']
            del kwargs['default']

        pars = {
            'fg': supports.__cp__['fg'],
            'font': 'Arial 12 bold'
        }

        for k in pars:
            if k not in kwargs:
                kwargs[k] = pars[k]

        super().__init__(parent, *args, **kwargs)

        self.__check_state = tk.BooleanVar(parent, default)
        self.__hover_cache = {'state': self.get()}  # construct dict for caching for style changes on hover
        self.__name = self['text']
        self.__mouse_down = tk.BooleanVar(parent, False)
        self.__fg = self['fg']

        # bind mouse actions
        self.bind('<Button-1>', self.__toggle)
        self.bind('<Enter>', self.__on_enter)
        self.bind('<Leave>', self.__on_leave)

        self.trace_add('write', self.__style_change)
        self.__style_change()  # update the style according to match the inputted default state

    def get(self):
        return self.__check_state.get()

    def set(self, value):
        self.__check_state.set(value)

    def __toggle(self, e):
        """Internal method that toggles the state of the checkbutton."""
        if self.get() is True:  # set state to off if on
            self.set(False)
        else:  # set state to on if off
            self.set(True)

    def __on_style(self):
        """Internal method that sets the checkbutton style when on."""
        self['fg'] = self.__fg

    def __off_style(self):
        """Internal method that sets the checkbutton style when off."""
        self['fg'] = supports.__cp__['disabled_bg']

    def __style_change(self, *_):
        """Internal method that changes the style based on the state of the checkbutton."""
        if self.get() is True:
            self.__on_style()
        else:
            self.__off_style()

    def __on_enter(self, e):
        """Internal method that changes the style on mouse-over."""
        self.__hover_cache['fg'] = self['fg']
        self.__hover_cache['state'] = self.get()
        self['fg'] = supports.highlight(self.__fg, -40)
        self['cursor'] = 'hand2'

    def __on_leave(self, e):
        """Internal method that changes the style on mouse-out."""
        # only reset style from cache if the state was not changed during hover
        if self.get() == self.__hover_cache['state']:
            self['fg'] = self.__hover_cache['fg']

    def trace_add(self, *args, **kwargs):
        self.__check_state.trace_add(*args, **kwargs)


class JSONVar(tk.StringVar):
    """Alteration of the StringVar method that allows for continuous storing of JSON convertible properties."""

    def __init__(self, parent, value: any, *args, **kwargs):
        value = json.dumps(value)
        super().__init__(parent, value, *args, **kwargs)

    def set(self, value: any):
        string = json.dumps(value)
        super().set(string)

    def get(self) -> any:
        string = super().get()
        return json.loads(string)


class DictVar(JSONVar):
    """Extension of the JSONVar that adds specific functionality for dictionary variables."""

    def __init__(self, parent, value=None, *args, **kwargs):
        value = {} if value is None else value
        super().__init__(parent, value, *args, **kwargs)

    def __setitem__(self, key, value):
        d = self.get()
        d[key] = value
        self.set(d)

    def __getitem__(self, item):
        return self.get()[item]

    def __delitem__(self, key):
        d = self.get()
        del d[key]
        self.set(d)

    def __contains__(self, item):
        d = self.get()
        if item in d: return True;
        else: return False;

    def update(self, d):
        _ = self.get()
        _ = supports.dict_update(_, d)
        self.set(_)


class DirectoryEntry(AppEntry):
    """Internal class that allows user to select a directory with a finder pop-up. The class will catch a cancelled
    operation such that the current real directory is not replaced with an empty directory."""

    def __init__(self, parent, *args, **kwargs):

        # set default parameters
        pars = {
            'state': tk.DISABLED,
            'relief': 'flat',
            'font': 'Arial 12',
            'width': 90,
            'forbidden': ()
        }

        for k in pars:
            if k not in kwargs:
                kwargs[k] = pars[k]

        self.forbidden_folders = kwargs['forbidden']
        del kwargs['forbidden']

        """Allow for a specific function call on active directory change. Note that this functionality differs from the 
        regular trace event 'write', as this functionality is only triggered upon user-prompted change of the directory. 
        It is NOT triggered by the .set() event, which would otherwise trigger the trace."""
        self.trigger_function = None
        if 'trigger_function' in kwargs:
            self.trigger_function = kwargs['trigger_function']
            del kwargs['trigger_function']

        super().__init__(parent, *args, **kwargs)

        self.parent = parent

        self.bind('<Button-1>', self.__ask_directory)  # bind the mouse click to action
        self.bind('<Enter>', self.__on_enter)

    def __ask_directory(self, e):
        directory = base.configure_directory(self.tkID, self.parent.tkID, forbidden=self.forbidden_folders)

        if directory is not None:
            self.set(directory)

        if self.trigger_function is not None:
            self.trigger_function()

    def __on_enter(self, e):
        self['cursor'] = 'hand2'


class _SelectionGridField(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.__state = tk.BooleanVar(self, value=False)
        self.__parent = parent
        self.__bg = kwargs['bg']

        self.bind('<Enter>', self.__on_enter)
        self.bind('<Leave>', self.__on_leave)
        self.bind('<Button-1>', self.__on_click)
        self.trace_add('write', self.__field_update)

    def __on_enter(self, e):
        self['bg'] = supports.highlight(self.__bg, 30)

    def __on_leave(self, e):
        self.__field_update(None)

    def __on_click(self, e):
        if self.get() is False:
            self.set(True)
        else:
            self.set(False)

    def get(self):
        return self.__state.get()

    def set(self, state):
        self.__state.set(state)

    def __field_update(self, *_):
        if self.get() is False:
            self['bg'] = self.__bg
        else:
            self['bg'] = supports.highlight(self.__bg, -40)

    def trace_add(self, *args, **kwargs):
        self.__state.trace_add(*args, **kwargs)


class SelectionGrid(AppFrame):
    def __init__(self, parent, grid, size=15, gap=1, fkwargs=None, mask=None, *args, **kwargs):

        _fkwargs = {
            'bg': supports.__cp__['dark_bg']
        }

        if fkwargs is None:
            fkwargs = _fkwargs
        else:
            for key, value in _fkwargs.items():
                fkwargs[key] = value

        super().__init__(parent, *args, **kwargs)

        self.__mask = mask
        self.__fields = {}
        self.selected_fields = JSONVar(self, value=[])
        self.display_fields = tk.StringVar(self, value='')
        for r in range(grid[0]):
            for c in range(grid[1]):
                self.__fields[(r, c)] = _SelectionGridField(self, width=size, height=size, **fkwargs)
                self.__fields[(r, c)].grid(row=r, column=c, padx=gap, pady=gap)
                self.__fields[(r, c)].trace_add('write', self.__set_selected_fields)
        self.grid_rowconfigure(0, minsize=size)
        self.grid_columnconfigure(0, minsize=size)

    def get(self):
        return self.selected_fields.get()

    def get_grid(self):
        returned_true = []
        for k, v in self.__fields.items():
            if v.get() is True:
                returned_true.append(k)
        return returned_true

    def set_grid(self, fields, value):
        for k in fields:
            if isinstance(k, list):  # fix tuple if it was loaded from json file
                k = tuple(k)
            if k in self.__fields:
                self.__fields[k].set(value)

    def get_grid_dict(self, invert=False, str_keys=False):
        """Method that returns all fields with their activity status as a dictionary."""
        _ = {}
        for k, v in self.__fields.items():
            if str_keys is True:  # convert key to string if prompted in call
                k = str(k)

            v = v.get()
            if invert is True:  # invert False/True statements for activity if prompted
                _[k] = False if v is True else True
            else:
                _[k] = v
        return _

    def reset_grid(self):
        for v in self.__fields.values():
            v.set(False)

    def __set_selected_fields(self, *_):
        """Internal method that sets the selected_fiels tkvariable"""
        mask = base.load_mask_file(self.__mask)
        _fields = [mask.iloc[i] for i in self.get_grid()]
        self.selected_fields.set(_fields)
        self.display_fields.set(str(_fields).removeprefix('[').removesuffix(']').replace("'", ''))


class WindowFrame(AppFrame):
    """Costum frame class for the cellexum application. This is meant to be used for child classing."""

    def __init__(self, parent, name=None, *args, **kwargs):

        pars = {'padx': 60,
                'pady': 30,
                'bg': supports.__cp__['bg']}
        for k in pars:
            if k not in kwargs:
                kwargs[k] = pars[k]

        super().__init__(parent, *args, **kwargs)

        self.__frame_counter = 0
        self.__frames = {}
        self.parent = parent

        if name is None:
            name = self.__name__

        self.name = name
        self.tkID = name
        self.__grid_kwargs = ('column', 'columnspan', 'ipadx', 'ipady', 'padx', 'pady', 'row', 'rowspan', 'sticky',
                              'autorow', 'prow', 'arc', 'ars')  # define allowed kwargs for the grid call
        self.last_frame = None
        self.__reserved_rows = {}
        self.settings = None
        self.containers = {}
        self.tags = {}
        self.groups = {}
        self.load_traces = True

        self.bind('<Destroy>', self._remove_class_traces)  # remove class traces on widget destruction

    def __getitem__(self, item):
        return self.__frames[item]

    def __setitem__(self, key, value):
        self.__frames[key] = value

    def __init_widget(self, wtype, sid, overwrite=False):

        if sid is None:
            _id = f'{wtype}{self.__frame_counter}'
        else:
            _id = f'{wtype}{sid}'

        if _id in self.__frames:
            if overwrite is True:
                self.__frames[_id].destroy()
                del self.__frames[_id]
            else:
                raise ValueError(f'Widget with id {_id} already exists.')

        self.__frame_counter += 1
        return _id

    def add(self, widget, sid=None, tag=None, group=None, tooltip=None, **kwargs):
        """Method to add widgets to the WindowFrame.
        :param widget: widget to add. This must be a base class from skeletons or classes derived thereof.
        :param sid: id of the widget to add. If 'text' is added and sid is None, the text will be used instead.
        :param tag: add a tag to the widget for iterative processing later on. Tags should be reserved for only 1D
        variation.
        :param group: add a group to the widget for iterative processing later on. Groups should be reserved for 2D
        variation or more."""

        # fetch hide kwarg if it is passed
        hide = False
        if 'hide' in kwargs:
            hide = kwargs['hide']; del kwargs['hide']

        # solve the sid
        if 'text' in kwargs and sid is None:
            sid = kwargs['text'].replace(' ', '')

        # solve kwargs and release an id for the widget to be added
        container, _grid, _widget, _init = self.__sort_kwargs(**kwargs)
        _id = self.__init_widget(widget.__name__, sid, **_init)

        # solve tags and groups
        if tag is not None:
            if tag not in self.tags:
                self.tags[tag] = set()
            self.tags[tag].add(_id)
        if group is not None:
            if group not in self.groups:
                self.groups[group] = set()
            self.groups[group].add(_id)

        warning = None
        if 'warning' in _widget:
            if 'command' in _widget:
                warning = _widget['warning']
                command = _widget['command']
                del _widget['warning']
                del _widget['command']

        self.__frames[_id] = elem = widget(container, **_widget)  # load widget

        elem.grid(**self.__set_grid(**_grid))  # place widget
        elem.tkID = _id  # set the tkinter ID as the created ID
        elem.tag = tag; elem.group = group
        if tooltip is not None:
            if tooltip is True:
                elem.tooltip(supports.__tt__[self.tkID][_id])
            else:
                elem.tooltip(tooltip)
        if container.tkID not in self.containers:
            self.containers[container.tkID] = []
        self.containers[container.tkID] += [elem.tkID]

        if warning is not None:
            if warning is True:
                warning = supports.__warnings__[self.tkID][_id]
            warning += '\n\nDo you want to continue?\n\n(control-click to suppress prompts)'

            def warning_command(e):
                def _():
                    prompt = messagebox.askokcancel('Warning', warning)
                    if prompt is True:
                        return command()

                elem.configure(command=_)

            def regular_command(e):
                supports.tprint('Suppressed warning.')
                elem.configure(command=command)

            elem.bind('<Button-1>', warning_command, '+')
            elem.bind('<Control-1>', regular_command, '+')

        if hide is True:
            elem.grid_remove()

        return elem

    def __set_grid(self, **kwargs):
        """Internal method that sets up the grid args and kwargs based on kwargs."""

        pars = {  # define standard parameters
            'autorow': self.name,
            'column': 0,
            'sticky': 'nw'
        }

        for k in pars:  # add standard parameters if they are not in the call
            if k not in kwargs:
                kwargs[k] = pars[k]

        if 'prow' in kwargs:  # add an argument to pause the row counter
            if kwargs['prow'] is True:
                kwargs['autorow'] = '@' + kwargs['autorow']
            del kwargs['prow']

        return kwargs

    def __sort_kwargs(self, **kwargs):
        """Internal method that sorts kwargs according to whether they belong to grid placement or setting kwarg."""
        grid_kwargs, setting_kwargs, initial_kwargs = {}, {}, {}

        if 'container' in kwargs:
            container = self[kwargs['container']]
            kwargs['autorow'] = kwargs['container']
            del kwargs['container']
        else:
            container = self

        if 'overwrite' in kwargs:
            initial_kwargs['overwrite'] = kwargs['overwrite']
            del kwargs['overwrite']

        for k in kwargs:  # sort kwargs
            if k in self.__grid_kwargs:
                grid_kwargs[k] = kwargs[k]
            else:
                setting_kwargs[k] = kwargs[k]
        return container, grid_kwargs, setting_kwargs, initial_kwargs

    def is_empty(self, container=None):
        """Method that checks whether the widget is empty."""
        if container is None:
            if not self.__frames:
                return True
            else:
                return False
        else:
            if not self.containers[container]:
                return True
            else:
                return False

    def __traces__(self):
        """Placeholder for global traces. To be loaded only after __base__ call and only once for the WindowFrame
        lifetime."""
        pass

    def load(self):
        """Method that loads in the widget if the base is empty"""
        try:
            if self.is_empty():  # base-load widget if it does not exist
                self.__base__()
            else:  # otherwise just place the widget
                self.grid()

            if self.load_traces is True:
                self.__traces__()
                self.load_traces = False
        except Exception as e:
            supports.tprint(f'Failed to load {self.tkID} with exit: {e!r}')
            if self.dv_get('Debugger') is True:
                raise e

    def _remove_class_traces(self, e):
        """Change native behavior to also drop all traces from the class."""
        self.remove_class_traces()

    def __base__(self):
        """Internal placeholder"""
        pass

    def clear(self, *_):
        """Method that clears the widget base."""
        for v in self.__frames.values():
            v.destroy()
        self.__frame_counter = 0
        self.__frames = {}
        self.last_frame = None
        self.__reserved_rows = {}
        self.settings = None
        self.containers = {}
        self.tags = {}
        self.groups = {}

    def reload(self, *_):
        """Method that reloads the widget base."""
        self.clear()
        try:
            self.__base__()
        except Exception as e:
            supports.tprint(f'Failed to load {self.tkID} with exit: {e!r}')
            if self.dv_get('Debugger') is True:
                raise e

    @property
    def last_frame_id(self):
        """Method that returns the latest added frame id"""
        return self.__frames.keys()[-1]

    def reserve_rows(self, n, name):
        """Method that reserves n amount of rows for future use under name."""
        if n > 0:
            self.__reserved_rows[name] = [auto_row(self.name) for _ in range(n)]
        else:
            self.__reserved_rows[name] = row_counter[self.name]

    def collect_rows(self, name):
        """Method that collects reserved rows under name."""
        return self.__reserved_rows[name]

    def load_settings(self, repeat=None):
        """Method that loads the Settings.json from the set output directory. Specifically, the function reads the json
        for all loaded repeats, unless repeat is set to either a repeat integer or the Repeat:# structure."""

        if isinstance(repeat, int):
            repeats = [f'Repeat:{repeat}']
        elif isinstance(repeat, str):
            repeats = [repeat]
        else:
            repeats = self.dv_get('RepeatData')

        _ = {}
        for r in repeats:
            _out = self.dv_get('OutputFolder', group='Directories', branch=r)
            json_path = rf'{_out}\Settings.json'
            _[r] = supports.json_dict_push(json_path, behavior='read')
        self.settings = _

    def fetch_ids(self):
        """Method that fetches all produced widget ids."""
        return self.__frames.keys()

    def container_drop(self, container, dtype='destroy', raise_error=False, kill=False):
        """Method that allows for dropping all content in a container without removing the container itself. Note that
        this method differs from the regular drop method in that it does not remove the target element, only the
        children in the container.
        :params container: the container for which elements should be dropped.
        :params dtype: the drop type. If 'destroy' the contained elements are destroyed and removed from the container
        entry. If 'remove' the elements are simply removed from the GUI to be recalled later. The default is 'destroy'.
        :params kill: If set to True, the container itself will be destroyed and cleaned from the container dict.
        """

        if container in self.containers:
            if dtype == 'destroy':
                for elem in self.containers[container]:
                    self.drop(elem)  # safely destroy tkinter widget
                self.containers[container] = []  # reset the container entry
            elif dtype == 'remove':  # this functionality is very niche - consider removing
                for elem in self.containers[container]:
                    self[elem].grid_remove()
            if kill:
                del self.containers[container]
                self.drop(container)
        else:
            if raise_error:
                raise ValueError(f'Container {container!r} does not exist.')

    def drop(self, elem):
        """Method that destroys a widget and its entries."""
        if elem in self.__frames:
            # check and remove element from tags and groups
            tag = self[elem].tag; group = self[elem].group
            if tag is not None: self.tags[tag].remove(elem);
            if group is not None: self.groups[group].remove(elem);

            # remove the element from tkinter and internally
            self[elem].destroy()
            del self.__frames[elem]

        if elem in self.containers:  # ensure proper child removal if the element is a container
            for _ in self.containers[elem]:
                try:
                    self.drop(_)  # remove all contained elements as well
                except KeyError:
                    pass
            del self.containers[elem]

    def exists(self, elem):
        """Method to check whether a tkID exists within the object."""
        if elem in self.__frames:
            return True
        else:
            return False


class _EntryGridField(SettingEntry):
    def __init__(self, parent, *args, **kwargs):

        pars = {
            'justify': 'center',
            'width': 2,
            'font': ('Courier New', 10),
            'bg': supports.highlight(supports.__cp__['dark_bg'], -30),
            'toggle': False
        }

        for k in pars:
            if k not in kwargs:
                kwargs[k] = pars[k]

        self.__toggle = kwargs['toggle']
        del kwargs['toggle']

        super().__init__(parent, *args, **kwargs)

        self.__parent = parent
        self.__bg = self['bg']
        self.toggle_memory = self.get()

        self.bind('<Enter>', self.__on_enter)
        self.bind('<Leave>', self.__on_leave)
        self.trace_add('write', self.__field_update)

        if self.__toggle is True:
            self.bind('<Control-1>', self.__state_toggle)

    def __on_enter(self, e):
        self['bg'] = supports.highlight(self.__bg, 30)

    def __on_leave(self, e):
        self.__field_update(None)

    def __field_update(self, *_):
        _val = self.get()
        if _val == '':
            self['bg'] = self.__bg
        else:
            self['bg'] = supports.highlight(self.__bg, 30)

        _size = len(_val)
        if _size < 2:
            self['width'] = 2
        else:
            self['width'] = _size + 1

    def __state_toggle(self, e):
        """Internal method that toggles the state of the entry."""
        if self['state'] in ('disabled', tk.DISABLED):
            self['state'] = 'normal'
            self.set(self.toggle_memory)
        else:
            self['state'] = 'disabled'
            self.toggle_memory = self.get()
            self.set('N/A')


class EntryGrid(AppFrame):
    def __init__(self, parent, grid=None, vargrid=None, gap=1, fkwargs=None, *args, **kwargs):

        _fkwargs = {}

        if fkwargs is None:
            fkwargs = _fkwargs
        else:
            for key, value in _fkwargs.items():
                fkwargs[key] = value

        super().__init__(parent, *args, **kwargs)

        self.__fields = {}
        self.__fkwargs = fkwargs
        self.__grid = None
        self.__variable_grid = False
        self.__grid_gap = gap
        if vargrid is not None:
            self.__grid = vargrid
            self.__variable_grid = True
        if grid is not None:
            self.__grid = grid

        self.setup_grid(None)  # set up grid initially

    def setup_grid(self, *_):
        if self.__variable_grid is True:  # fetch grid according to type
            grid = []
            for e in self.__grid:
                _val = e.get()
                if _val == '':  # exit prematurely, if the input is blank, i.e. being written
                    return
                else:
                    grid.append(int(_val))
        else:
            grid = self.__grid

        valid_fields = []
        for r in range(grid[0]):  # construct a grid of entry fields, respecting existing entries
            for c in range(grid[1]):
                if (r, c) not in self.__fields:
                    self.__fields[(r, c)] = _EntryGridField(self, **self.__fkwargs)
                    self.__fields[(r, c)].grid(row=r, column=c, padx=self.__grid_gap, pady=self.__grid_gap,
                                               sticky='nswe')
                valid_fields.append((r, c))

        # remove fields that should no longer exist
        invalid_fields = [k for k in self.__fields if k not in valid_fields]
        for k in invalid_fields:
            self.__fields[k].destroy()
            del self.__fields[k]

    def get_grid(self):
        """Method that allows for fetching the entire grid as a dict of indices and grid element values."""
        _ = {}
        for k, v in self.__fields.items():
            _[k] = v.get()
        return _

    def set_grid(self, setting):
        """Method that allows for setting the entire grid.
        :param setting: a dict containing grid indices as keys and grid element values as values."""
        for k, v in setting.items():
            self.__fields[k].set(v)  # set the grid value

    def reset_grid(self):
        """Method that allows for resetting the entire grid values."""
        for v in self.__fields.values():
            v.set('')

    def get_grid_state(self, str_keys=False):
        """Method that allows for fetching the state of each element in the grid. If a field is active, the field value
        is True, otherwise it is False."""
        _ = {}
        for k, v in self.__fields.items():
            if str_keys is True:
                k = str(k)
            if v['state'] in ('normal', tk.NORMAL):
                _[k] = True
            else:
                _[k] = False
        return _

    def set_grid_state(self, states):
        """Method that allows for setting the state of each element in the grid."""
        for k, v in states.items():
            if isinstance(k, str):
                k = base.str_to_tuple(k)
            if isinstance(k, list):
                k = tuple(k)
            if isinstance(k, tuple):
                pass
            else:
                raise ValueError(f'Invalid intype: {type(k)!r}')
            if v is True:
                if self.__fields[k]['state'] not in ('normal', tk.NORMAL):
                    self.__fields[k]['state'] = 'normal'
                    self.__fields[k].set(self.__fields[k].toggle_memory)
            else:
                if self.__fields[k]['state'] not in ('disabled', tk.DISABLED):
                    self.__fields[k]['state'] = 'disabled'
                    self.__fields[k].toggle_memory = self.__fields[k].get()
                    self.__fields[k].set('<NA>')


class PopupWindow(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.winfo_toplevel().protocol('WM_DELETE_WINDOW', self.cancel_window)  # add cancel window protocol

    def cancel_window(self):
        """Internal method that destroys popup."""
        self.winfo_toplevel().destroy()


class MaskCreatorWindow(PopupWindow):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # load placeholder binds
        self.bind('<Control-i>', self.import_mask_bind)
        self.bind('<Control-o>', self.open_mask_bind)
        self.bind('<Control-s>', self.save_mask_bind)

    def fetch_mask(self, name_entry_sid=None, **kwargs):
        """Method that gets a mask path for a selected mask file and presets a mask name. Kwargs are passed
        to the askopenfile call.
        :params name_entry_sid: The sid of the name entry frame within the window. If None, there is no value being set.
        """
        file = filedialog.askopenfile(parent=self.parent, **kwargs)
        if not file:
            return  # add exit statement
        path = file.name  # get mask path
        file.close()  # exit opened file
        file_name = '.'.join(path.split(r'/')[-1].split('.')[:-1])
        if name_entry_sid is not None:
            self[name_entry_sid].set(file_name)  # preset mask name
        return path, file_name

    @staticmethod
    def import_mask(path):
        """Internal method that imports and sets a mask."""
        mask_df = base.import_mask(path)
        mask_dict = base.strip_dataframe(mask_df)  # convert mask
        return mask_df, mask_dict

    def import_mask_bind(self, e):
        pass

    def save_mask_bind(self, e):
        pass

    def open_mask_bind(self, e):
        pass


class LoadingCircle(AppFrame):
    def __init__(self, parent, *args, **kwargs):

        pars = {
            'aai': 5,
            'size': 32,
            'width': 4.5,
            'stepsize': .5,
            'delay': 10,
            'outline': '#ffffff',
            'style': 'arc',
            'extent': 320,
            'start': 0,
        }

        for p in pars:  # update custom parameters if needed
            if p in kwargs:
                pars[p] = kwargs[p]
                del kwargs[p]

        if 'bg' not in kwargs:
            kwargs['bg'] = '#474747'

        self._onset = pars['start']
        del pars['start']

        super().__init__(parent, *args, **kwargs)

        self.active = tk.BooleanVar(self, False)
        self.i = 0

        self.canvas = tk.Canvas(self, width=pars['size'], height=pars['size'], highlightthickness=0, **kwargs)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.coords = (pars['width'], pars['width'], pars['size'] - pars['width'], pars['size'] - pars['width'])
        self.__pars = pars
        self.__aa_gradient = supports.ColorGradient(onset=kwargs['bg'], end=pars['outline'])

    def __loading_loop(self):
        """Internal method that starts the loop cycle."""
        if self.get() is False:
            return

        self.i += 1

        # reset canvas and add rotated arc with a cycle delay
        self.canvas.delete('all')
        self.antialias_arc(start=self._onset + self.i, **self.__pars)
        self.after(self.__pars['delay'], self.__loading_loop)

    def antialias_arc(self, aai=3, stepsize=.5, **kwargs):
        """Method that creates fake anti-aliased arcs."""
        for _ in ('delay', 'size'):  # clean up kwargs
            del kwargs[_]

        if aai == 1:
            self.canvas.create_arc(self.coords, **kwargs)
        else:
            for i in range(aai):
                kwargs['outline'] = self.__aa_gradient.gradient(i / (aai - 1))
                kwargs['width'] -= stepsize
                self.canvas.create_arc(self.coords, **kwargs)

    def start(self):
        self.active.set(True)
        self.after(200, self.__loading_loop)  # delay loading animation to avoid flashing it

    def pause(self):
        self.active.set(False)

    def stop(self):
        self.active.set(False)
        self.canvas.delete('all')
        self.i = 0

    def get(self):
        return self.active.get()


class TopLevelProperties:
    def __init__(self, dep_var=None):
        self.dependent_variables = dep_var if dep_var is not None else {}
        self.traces = {}

        try:  # load in default values if cache files do not exist
            self.defaults = supports.json_dict_push(rf'{supports.__cache__}\application.json',
                                                    behavior='read')['ApplicationSettings']
        except KeyError:
            defaults = supports.json_dict_push(rf'{supports.__cwd__}\defaults.json', behavior='read')
            supports.json_dict_push(rf'{supports.__cache__}\application.json', params=defaults, behavior='update')
            self.defaults = defaults['ApplicationSettings']

        self.bind('<Button-1>', self.clear_selection_menu)
        self.bind('<Escape>', self.clear_selection_menu)

    def clear_selection_menu(self, e):
        if 'ActiveSelectionMenu' in self.dependent_variables['']['']:
            if self.dependent_variables['']['']['ActiveSelectionMenu'].get() is True:
                self.dependent_variables['']['']['ActiveSelectionMenu'].set(False)


class TopLevelWidget(tk.Toplevel, TopLevelProperties):
    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, parent, *args, **kwargs)
        TopLevelProperties.__init__(self, dep_var={'': {'': {  # construct global variable dict
            'LastClick': tk.StringVar(self, '')
        }}})

        self.dependent_variables['']['']['TooltipTimer'] = tk.IntVar(self, self.defaults['TooltipTimer'])

        self.main = _ = ContentFrame(self)
        _.pack(fill='both', expand=True)

        self.bind('<Configure>', self.update_content_frame)

    def update_content_frame(self, e):
        self.main.canvas.config(height=self.winfo_height())
        self.main.refresh_content_frame()


class FieldMaskCreator(MaskCreatorWindow):
    def __init__(self, parent, tie, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.tie = tie
        self.sample_type = self.tie.parent['SelectionMenuSampleType'].get()

    def __save_mask(self):
        pass

    def cancel_window(self):
        """Update cancel window functionality to insert fallback setting."""
        # define fallback functionality to prevent the 'Add ...' selection to ever be selected as the mask
        current = self.tie['SelectionMenuImageMask'].get(); previous = self.tie['SelectionMenuImageMask'].previous
        if current == 'Add ...':
            if previous == 'Add ...':
                self.tie['SelectionMenuImageMask'].set(self.tie['SelectionMenuImageMask'].default)
            else:
                self.tie['SelectionMenuImageMask'].set(previous)
        super().cancel_window()

    def save_mask_bind(self, e):
        return self.__save_mask()


class BaseProcessingFrame(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.selected_repeat = 'Repeat:0'
        self.active_branches = {}
        self.activation_bypass = True
        self.active_repeats = []
        self._folder_change_tt = "Change output folder of selected repeat '%s'. It is currently set to '%s'."

    def hide_table_entry(self, file):
        for name in self.setting_package(file):
            if self.exists(name):
                self[name].grid_remove()

    def show_table_entry(self, file):
        exists = True
        for name in self.setting_package(file):
            if self.exists(name):
                self[name].grid()
            else:
                exists = False

        if not exists:
            self.add_table_entry(file)

    def remove_table_entry(self, file):
        for name in self.setting_package(file):
            if self.exists(name):
                self.drop(name)

    def update_repeat_selection(self, repeat):
        self.selected_repeat = repeat
        for b, branch in self.active_branches.items():
            for k in branch:
                if b == repeat:
                    self.show_table_entry(rf'{b}:{k}')  # ensure that all files exist in the widget space
                else:
                    self.hide_table_entry(rf'{b}:{k}')

        if self.exists('ImageButtonChangeFolder'):  # update change folder button tooltip
            self._update_change_folder_button_tt(repeat)

    def _update_change_folder_button_tt(self, repeat):
        r_name = self.dv_get('RepeatData')[repeat]['Name']
        _out = self.dv_get('OutputFolder', group='Directories', branch=repeat)
        self['ImageButtonChangeFolder'].tooltip(self._folder_change_tt % (r_name, _out))

    def update_repeats(self, *_):
        """Internal method that updates the repeat functionality."""
        repeats = self.dv_get('RepeatData')
        if self.active_repeats != list(repeats.keys()) or self.activation_bypass is True:
            self['AppFrameRepeatContainer'].grid_remove()

            _ = {}; _active_repeats = []
            for k, v in repeats.items():
                _[v['Name']] = partial(self.update_repeat_selection, k)
                if k not in self.active_repeats:
                    _active_repeats.append(k)

            if not self.exists('SelectionMenuRepeatSelection'):
                self.add(AppLabel, text='Repeat: ', sid='RepeatSelection', container='AppFrameRepeatContainer')
                self.add(SelectionMenu, options=[], sid='RepeatSelection', container='AppFrameRepeatContainer',
                         prow=True, column=1, padx=(5, 0))

                self.add(ImageButton, sid='ChangeFolder', image='folder', command=self.change_output_folder,
                         container='AppFrameRepeatContainer', bg=supports.__cp__['bg'],
                         hover=supports.__cp__['dark_bg'], prow=True, column=2, padx=(10, 0), pady=(5, 0))
                self._update_change_folder_button_tt(self.selected_repeat)  # add tooltip

            options = tuple(_)
            self['SelectionMenuRepeatSelection'].update_options(options, commands=_, default=options[0], force=True)

            if len(repeats) > 1:
                self['AppFrameRepeatContainer'].grid()
                self.active_repeats = _active_repeats
            else:  # update default directory entry to match repeat structure
                self.active_repeats = []

        self.activation_bypass = False  # this ensures that the function is always run when the tab is first loaded in

    def change_output_folder(self):
        _forbidden = self.dv_get('InputFolder', group='Directories', branch=None, as_dict=True)
        which = self.selected_repeat
        directory = base.configure_directory(self.tkID, which, forbidden=_forbidden.keys(),
                                             forbidden_message='Selected directory is a configured input folder.')

        if directory:
            # update entries and cache files
            """There is something else this needs to update as well???"""
            _ = {_forbidden[which]: {'OutputFolder': directory}}
            supports.json_dict_push(rf'{supports.__cache__}\saves.json', _, behavior='update')

            _ = {'DirectorySettings': {which: {'OutputFolder': directory}}}
            supports.json_dict_push(rf'{supports.__cache__}\settings.json', _, behavior='update')

            self.dv_set('OutputFolder', directory, group='Directories', branch=which)
            self._update_change_folder_button_tt(self.selected_repeat)  # update tooltip during directory change

    @staticmethod
    def setting_package(file): return [];

    def add_table_entry(self, file): pass;

    @staticmethod
    def split_file_name(file, r=str):
        file_repeat, file_name = file.split(':')[1:]
        file_repeat = file_repeat if r is int else f'Repeat:{file_repeat}'
        return file_repeat, file_name


class FieldProcessingFrame(BaseProcessingFrame):
    def __init__(self, parent, *args, **kwargs):
        kwargs['padx'] = 0; kwargs['pady'] = 0
        super().__init__(parent, *args, **kwargs)

        self.files = []

    def update_image_settings(self, *_):
        """Internal method that updates setting table."""
        branches = self.dv()['SelectedFiles'].keys()
        _ = []; active_branches = {}
        for b in branches:
            branch_files = self.dv_get(group='SelectedFiles', branch=b, as_dict=True)
            for k, v in branch_files.items():
                self.show_table_entry(rf'{b}:{k}')  # ensure that all files exist in the widget space
                if v is True:
                    _.append(rf'{b}:{k}')
                else:
                    self.hide_table_entry(rf'{b}:{k}')

                try:
                    if len(branches) > 1:
                        self[rf'TextCheckbutton:{b}:{k}']['text'] = rf'{k} {self.dv("RepeatData")[b]["Uncommon"]}'
                    else:
                        self[rf'TextCheckbutton:{b}:{k}']['text'] = rf'{k}'
                except KeyError:
                    self[rf'TextCheckbutton:{b}:{k}']['text'] = rf'{k}'
            active_branches[b] = branch_files.keys()  # add branch as active
        self.files = _

        for branch, files in self.active_branches.items():
            if branch not in active_branches:
                for k in files:
                    try:
                        self.remove_table_entry(rf'{branch}:{k}')  # drop entry if its branch has been removed
                    except KeyError: pass;

        self.active_branches = active_branches  # update the active branches

    def select_missing(self):
        """Attempt to determine which files are yet to be preprocessed and selected those files for processing."""
        repeat_out = self.dv_get('OutputFolder', group='Directories', branch=self.selected_repeat)
        try:
            processed = [i.removesuffix('.png') for i in os.listdir(rf'{repeat_out}\_masks for manual control') if
                         i.endswith('.png')]
        except FileNotFoundError:
            processed = []

        for f in self.dv()['SelectedFiles'][self.selected_repeat]:
            if f not in processed:
                self[f'TextCheckbutton:{self.selected_repeat}:{f}'].set(True)
            else:
                self[f'TextCheckbutton:{self.selected_repeat}:{f}'].set(False)

    def update_all(self, e):
        """Internal method that updates all table settings in a column if the last clicked widget was a setting."""
        last_click = self.dv_get('LastClick')
        if last_click in self.groups['ImageSettingsTable']:
            stype = last_click.split(':')[0]
            value = self[last_click].get()
            for file in self.files:
                self[rf'{stype}:{file}'].set(value)

    def restore_default(self):
        for option in self.groups['ImageSettingsTable']:
            if option.split(':')[1:3] == self.selected_repeat.split(':'):  # reset only selected repeat
                self[option].set(self[option].default)

    def update_image_mask_cache(self, *_):
        value = self['SelectionMenuImageMask'].get()
        if value not in ('Select Option', 'Add ...'):
            supports.post_cache({'PreprocessingSettings': {'ImageMask': value}})

    def preprocess_check(self) -> bool:
        _continue = True
        if self['SelectionMenuImageMask'].get() == 'Select Option':
            messagebox.showerror('Error', "Select an image mask before continuing.", )
            _continue = False
        if self.parent['SelectionMenuMaskChannel'].get() == 'Select Option':
            messagebox.showerror('Error', "Select a mask channel before continuing.", )
            _continue = False
        return _continue

    @supports.thread_daemon
    def preprocess_daemon(self):
        files = [file for file in self.files if self[f'TextCheckbutton:{file}'].get() is True]  # get active files
        base.RawImageHandler().handle(files)  # handle all images before preprocessing

        counter = 1; total = len(files)
        with concurrent.futures.ProcessPoolExecutor(max_workers=supports.get_max_cpu(),
                                                    mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(base.PreprocessingHandler().preprocess, file): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                file_repeat, file_name = file.split(':')[1:]
                uncommon = self.dv_get("RepeatData")[f"Repeat:{file_repeat}"]["Uncommon"]
                uncommon = f' {uncommon}' if uncommon is not None else ''

                try:
                    wpars = future.result()
                    supports.json_dict_push(r'{}\Settings.json'.format(
                        self.dv_get('OutputFolder', group='Directories', branch=f'Repeat:{file_repeat}')), wpars)
                    supports.tprint(f'Preprocessed image {file_name}{uncommon} ({counter}/{total}).')
                    counter += 1
                    time.sleep(.5)  # avoid overlapping instances that may overload the GUI modules
                    self.dv_set('LatestPreprocessedFile', file)
                except Exception as exc:
                    supports.tprint(f'Failed to preprocess {file_name}{uncommon} with exit: {exc!r}')
                    if self.dv_get('Debugger') is True:
                        raise exc
        supports.tprint('Completed all preprocessing.')

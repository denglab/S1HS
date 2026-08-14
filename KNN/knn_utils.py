import copy
import gc
import inspect
import json
import math
import os
import pickle
import platform
import random as rd
import re
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Union, Generator, Callable
from datetime import date
from time import ctime, time

import dill
import h5py

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
from matplotlib import rc, rcParams, tri
from scipy.io import loadmat


def is_interactive():
    """
    https://stackoverflow.com/a/39662359/622119
    """
    from IPython import get_ipython

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other types
    except NameError:
        return False  # Probably standard Python interpreter


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " x ".join(map(str, size))


def get_size(bytes, suffix="B"):
    """
    by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MiB'
        1253656678 => '1.17GiB'
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(bytes) < 1024.0:
            return f"{bytes:3.2f} {unit}{suffix}"
        bytes /= 1024.0
    return f"{bytes:3.2f} 'Yi'{suffix}"


def get_file_size(filename):
    file_size = os.stat(filename)
    return get_size(file_size.st_size)


def get_lambda_str(f):
    return (str(inspect.getsourcelines(f)[0])).strip("['\\n']").split(" = ")[1]


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.decode("utf-8").split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*: ", "", line, 1)


def get_system():
    print("=" * 40, f"{'CPU': <5}", "=" * 40)
    # number of cores
    print(f"Device name       : {get_processor_name()}")
    print(f"Physical cores    : {psutil.cpu_count(logical=False)}")
    print(f"Total cores       : {psutil.cpu_count(logical=True)}")
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency     : {cpufreq.max:.2f} Mhz")
    print(f"Min Frequency     : {cpufreq.min:.2f} Mhz")
    print(f"Current Frequency : {cpufreq.current:.2f} Mhz")

    if torch.cuda.is_available():
        print("=" * 40, f"{'GPU': <5}", "=" * 40)
        print(f"{'Device': <18}: {torch.cuda.get_device_name(0)}")
        print(
            f"{'Mem total': <18}: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB"
        )
        print(
            f"{'Mem allocated': <18}: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB"
        )
        print(
            f"{'Mem cached': <18}: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB"
        )
    print("=" * 40, f"{'Memory': <8}", "=" * 40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"{'Total': <18}: {get_size(svmem.total)}")
    print(f"{'Available': <18}: {get_size(svmem.available)}")
    print(f"{'Used': <18}: {get_size(svmem.used)}")

    print("=" * 40, f"{'Software': <8}", "=" * 40)
    print(f"{'Python': <18}: {sys.version.split(' (')[0]}")
    print(f"{'Numpy': <18}: {np.__version__}")
    print(f"{'PyTorch': <18}: {torch.__version__}")

    print("=" * 40, "system info print done", "=" * 40)


def get_seed(s, quiet=True, cudnn=True, logger=None):
    # rd.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    message = f""""""
    message += f"""
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    """
    if cudnn:
        message += f"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""

    if torch.cuda.is_available():
        message += f"""
    torch.cuda.manual_seed_all({s})"""
        
    if not quiet and not logger:
            print("\n")
            print(f"The following code snippets have been run.")
            print("=" * 50)
            print(message)
            print("=" * 50)
    elif not quiet and logger:
        logger.info(
            "The following code snippets have been run:"
            + " | ".join(message.splitlines())
        )


class Colors:
    """Defining Color Codes to color the text displayed on terminal."""

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


@contextmanager
def timer(label: str = "", verbose=False, quiet=False) -> Generator[None, None, None]:
    """
    https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203020#1111022
    print
    1. the time the code block takes to run
    2. the memory usage.
    """
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2.0**30
    start = time()  # Setup - __enter__
    if verbose and not quiet:
        print(color(f"{label}:\nStart at {ctime(start)};", color=Colors.blue))
        try:
            yield  # yield to body of `with` statement
        finally:  # Teardown - __exit__
            m1 = p.memory_info()[0] / 2.0**30
            delta = m1 - m0
            sign = "+" if delta >= 0 else "-"
            delta = math.fabs(delta)
            end = time()
            print(
                color(
                    f"Done  at {ctime(end)} ({end - start:.6f} secs elapsed);",
                    color=Colors.blue,
                )
            )
            print(color(f"\nLocal RAM usage at START: {m0:.2f} GB", color=Colors.green))
            print(
                color(
                    f"Local RAM usage at END:   {m1:.2f}GB ({sign}{delta:.2f}GB)",
                    color=Colors.green,
                )
            )
            print("\n")
    elif not verbose and not quiet:
        yield
        print(
            color(
                f"{label} - done in {time() - start:.6f} seconds. \n", color=Colors.blue
            )
        )
    else:
        try:
            yield
        finally:
            pass


def get_memory(num_var=10):
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in globals().items()),
        key=lambda x: -x[1],
    )[:num_var]:
        print(
            color(f"{name:>30}:", color=Colors.green),
            color(f"{get_size(size):>8}", color=Colors.magenta),
        )


def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for _file in files:
            if name in _file:
                result.append(os.path.join(root, _file))
                print(f"Found {_file}")
    return sorted(result)


def print_file_size(files):
    for f in files:
        size = get_file_size(f)
        filename = f.split("/")[-1]
        filesize = get_file_size(f)
        print(
            color(f"{filename:>30}:", color=Colors.green),
            color(f"{filesize:>8}", color=Colors.magenta),
        )


def pretty_tensor_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " x ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s:%s%s %s"
                        % (
                            type(obj).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.is_pinned else "",
                            pretty_tensor_size(obj.size()),
                        )
                    )
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    info = (
                        f"{type(obj).__name__} → {type(obj.data).__name__}" + " GPU"
                        if obj.is_cuda
                        else (
                            "" + " pinned"
                            if obj.data.is_pinned
                            else (
                                "" + " grad"
                                if obj.requires_grad
                                else (
                                    "" + " volatile"
                                    if obj.volatile
                                    else "" + f" {pretty_tensor_size(obj.data.size())}"
                                )
                            )
                        )
                    )
                    print(info)
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", get_size(total_size))


@contextmanager
def trace(title: str):
    t0 = time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2.0**30
    yield
    m1 = p.memory_info()[0] / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(
        f"[{m1:.1f}GB ({sign}{delta:.3f}GB): {time() - t0:.2f}sec] {title} ",
        file=sys.stderr,
    )


def get_cmap(n, cmap="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(cmap, n)


def get_date():
    today = date.today()
    return today.strftime("%b-%d-%Y")


def argmax(lst):
    """
    Taken from https://stackoverflow.com/a/31105620/622119
    License: CC BY-SA 3.0.
    """
    return lst.index(max(lst))


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        num_params += np.prod(p.size() + (2,) if p.is_complex() else p.size())
        # num_params += p.numel() * (1 + p.is_complex())
    return num_params


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df


def save_pickle(data, save_path, append=True):
    mode = "ab" if append else "wb"
    with open(save_path, mode) as f:
        dill.dump(data, f)

def load_pickle(load_path, mode="rb"):
    data = []
    with open(load_path, mode=mode) as f:
        try:
            while True:
                data.append(dill.load(f))
        except EOFError:
            pass
    if len(data) == 1:
        return data[0]
    else:
        return data

def save_dict_to_json(d, save_path):
    with open(save_path, "w") as f:
        json.dump(d, f, indent=4, sort_keys=True)


def read_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def default(value, d):
    """
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value


class DotDict(dict):
    """
    https://stackoverflow.com/a/23689767/622119
    https://stackoverflow.com/a/36968114/622119
    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def clones(module, N):
    """
    Input:
        - module: nn.Module obj
    Output:
        - zip identical N layers (not stacking)

    Refs:
        - https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def csr_to_sparse(M):
    '''
    Input:
        M: csr_matrix
    Output:
        torch sparse tensor

    Another implementation can be found in
    https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    '''
    n, m = M.shape
    coo_ = M.tocoo()
    ix = torch.LongTensor([coo_.row, coo_.col])
    M_t = torch.sparse.FloatTensor(ix, torch.from_numpy(M.data).float(), [n, m])
    return M_t


def showmesh(node, elem, **kwargs):
    triangulation = tri.Triangulation(node[:, 0], node[:, 1], elem)
    markersize = 3000 / len(node)
    if kwargs.items():
        h = plt.triplot(triangulation, "b-h", **kwargs)
    else:
        h = plt.triplot(
            triangulation, "b-h", linewidth=0.5, alpha=0.5, markersize=markersize
        )
    return h


def showsolution(node, elem, u, **kwargs):
    """
    show 2D solution either of a scalar function or a vector field
    on triangulations
    """
    markersize = 3000 / len(node)

    if u.ndim == 1:  # (N, )
        uplot = ff.create_trisurf(
            x=node[:, 0],
            y=node[:, 1],
            z=u,
            simplices=elem,
            colormap="Viridis",  # similar to matlab's default colormap
            showbackground=True,
            show_colorbar=False,
            aspectratio=dict(x=1, y=1, z=1),
        )
        fig = go.Figure(data=uplot)

    elif u.ndim == 2 and u.shape[1] == 2:  # (N, 2)
        if u.shape[0] == elem.shape[0]:
            u /= (np.abs(u)).max()
            node = node[elem].mean(axis=1)

        uplot = ff.create_quiver(
            x=node[:, 0],
            y=node[:, 1],
            u=u[:, 0],
            v=u[:, 1],
            scale=0.2,
            arrow_scale=0.5,
            name="gradient of u",
            line_width=1,
        )

        fig = go.Figure(data=uplot)

    if "template" not in kwargs.keys():
        fig.update_layout(
            template="plotly_dark", margin=dict(l=5, r=5, t=5, b=5), **kwargs
        )
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), **kwargs)
    fig.show()
    return fig


def showsurf(x, y, z, **kwargs):
    """
    show 2D solution either of a scalar function or a vector field
    on a meshgrid
    x, y, z: (M, N) matrix
    """

    uplot = (go.Surface(x=x, y=y, z=z, colorscale="Viridis", showscale=False),)

    fig = go.Figure(data=uplot)

    if "template" not in kwargs.keys():
        fig.update_layout(
            template="plotly_dark", margin=dict(l=5, r=5, t=5, b=5), **kwargs
        )
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), **kwargs)
    fig.show()


def showcontour(
    z,
    colorscale="RdYlBu",
    showscale=False,
    showlabels=False,
    continuous_coloring=False,
    reversescale=True,
    dimensions=(200, 200),
    line_smoothing=0.7,
    ncontours=20,
    **plot_kwargs,
):
    """
    show 2D solution z of its contour
    colorscale: balance (MATLAB new) or Jet (MATLAB old)
    """

    if not plot_kwargs:
        plot_kwargs = dict(
            contour_kwargs=dict(
                colorscale=colorscale,
                line_smoothing=line_smoothing,
                line_width=0.1,
                ncontours=ncontours,
                reversescale=reversescale,
                # )
            ),
            figure_kwargs=dict(
                layout={
                    "xaxis": {
                        "title": "x-label",
                        "visible": False,
                        "showticklabels": False,
                    },
                    "yaxis": {
                        "title": "y-label",
                        "visible": False,
                        "showticklabels": False,
                    },
                }
            ),
            layout_kwargs=dict(
                margin=dict(l=0, r=0, t=0, b=0),
                width=dimensions[0],
                height=dimensions[1],
                template="plotly_white",
            ),
        )

    contour_kwargs = plot_kwargs["contour_kwargs"]
    figure_kwargs = plot_kwargs["figure_kwargs"]
    layout_kwargs = plot_kwargs["layout_kwargs"]
    if showscale:
        contour_kwargs["showscale"] = True
        contour_kwargs["colorbar"] = dict(
            thickness=0.15 * layout_kwargs["height"],
            tickwidth=0.3,
            exponentformat="e",
        )
        layout_kwargs["width"] = 1.32 * layout_kwargs["height"]
        # layout_kwargs['coloraxis_colorbar'] = dict(thickness=0.32*layout_kwargs['height'])
    else:
        contour_kwargs["showscale"] = False

    if continuous_coloring:
        contour_kwargs["contours_coloring"] = "heatmap"

    if showlabels:
        contour_kwargs["contours"] = dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="gray",
            ),
        )

    uplot = go.Contour(z=z, **contour_kwargs)
    fig = go.Figure(data=uplot, **figure_kwargs)
    if "template" not in layout_kwargs.keys():
        fig.update_layout(template="plotly_dark", **layout_kwargs)
    else:
        fig.update_layout(**layout_kwargs)

    return fig


def get_config(module: nn.Module, quiet=True, logger=None):
    config = {}
    _config = filter(lambda x: not x.startswith("_"), dir(module))
    for a in _config:
        if not isinstance(getattr(module, a), (Callable, nn.Parameter, torch.Tensor)):
            config[a] = getattr(module, a)
    if not quiet and not logger:
        for k, v in config.items():
            print(f"{k:<25}: {v}")
    elif logger:
        logger.info(f"args of {module.__repr__()}: "+" | ".join(f"{k}={v}" for k, v in config.items()))
    return config
    


if __name__ == "__main__":
    get_system()
    get_memory()
else:
    with timer(f"Loading modules for visualization", quiet=True):
        try:
            import plotly.express as px
            import plotly.figure_factory as ff
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError as err:
            sys.stderr.write(f"Error: failed to import module ({err})")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])

        if is_interactive():
            try:
                import seaborn as sns

                sns.set_theme(style="darkgrid", context="talk")
                from jupyterthemes import jtplot

                jtplot.style(
                    theme="onedork", context="notebook", ticks=True, grid=False
                )
            except ImportError as err:
                sys.stderr.write(f"Error: failed to import module ({err})")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "seaborn", "jupyterthemes"]
                )


from hyperopt.pyll.base import scope, as_apply, Apply
from collections import defaultdict


param_names = {"randint", "uniform", "normal", "loguniform", "lognormal",
               "quniform", "qnormal", "qloguniform", "qlognormal"}

for param in param_names:
    globals()[param] = getattr(scope, param)


def is_float(rv_name): return rv_name != "randint"


def choice(*options):
    return scope.switch(scope.randint(len(options)), *options)


def pchoice(*options):
    p, options = zip(*p_options)
    n_options = len(options)
    ch = scope.categorical(p, upper=n_options)
    return scope.switch(ch, *options)


def label_vars(expr, path, suffixes=None):
    if suffixes is None: suffixes = defaultdict(lambda: 0)

    if expr.name in param_names:
        idx = suffixes[path]
        suffixes[path] += 1
        if idx == 0: label = path
        else: label = "%s_%d" % (path, idx)

        param = scope.hyperopt_param(label, expr)
        if is_float(expr.name):
            param = scope.float(param)
        return param

    elif expr.name == "dict":
        new_dict = {}
        for (key, value) in expr.named_args:
            new_dict[key] = label_vars(value, path + "." + key, suffixes)
        return scope.dict(**new_dict)

    # Default:
    inputs = [label_vars(v, path, suffixes) for v in expr.inputs()]
    return expr.clone_from_inputs(inputs)


def instance(cls, params):
    class_name = "%s.%s" % (cls.__module__, cls.__name__)
    kwargs = dict(params)
    kwargs["__class__"] = class_name
    return scope.dict(**kwargs)


def parameter(name, low=None, high=None, dist=None):
    if dist is not None:
        # We have a distribution already, but maybe the user specified
        # hard bounds in addition to it?
        if low is not None: dist = scope.max(low, dist)
        if high is not None: dist = scope.min(high, dist)
    else:
        # No distribution specified. Assume a uniform distribution and
        # use the low and high bounds:
        if low is None: low = 0
        if high is None:
            raise ValueError("high must be specified when dist is None.")
        dist = uniform(low, high)

    return dist


__all__ = ["label_vars", "choice", "pchoice"] + list(param_names)



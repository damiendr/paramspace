
from hyperopt.pyll.base import scope, as_apply
from hyperopt import hp
from collections import defaultdict


param_names = {"randint", "uniform", "normal", "loguniform", "lognormal",
               "quniform", "qnormal", "qloguniform", "qlognormal"}

for param in param_names:
    # hyperopt.pyll.base.scope defines the label-less forms of the above
    # hyperopt functions:
    globals()[param] = getattr(scope, param)


def bool(): return randint(2)


def choice(*options):
    # Like hp.choice, but without a label.
    return scope.switch(scope.randint(len(options)), *options)


def pchoice(*options):
    # Like hp.pchoice, but without a label.
    p, options = zip(*p_options)
    n_options = len(options)
    ch = scope.categorical(p, upper=n_options)
    return scope.switch(ch, *options)


def label_vars(expr, path="", suffixes=None, delim="."):
    if suffixes is None: suffixes = defaultdict(lambda: 0)

    if expr.name == "hyperopt_param":
        return expr # that one's already labeled.

    if expr.name in param_names:
        # We have a parameter. Let's find a label for it:
        idx = suffixes[path]
        suffixes[path] += 1
        if idx == 0: label = path
        else: label = "%s_%d" % (path, idx)

        # Now that we have the label, we can rebuild the node
        # using the corresponding method in hyperopt.hp:
        args = expr.pos_args
        kwargs = dict(expr.named_args)
        param = getattr(hp, expr.name)(label, *args, **kwargs)
        return param

    elif expr.name == "dict":
        new_dict = {}
        for (key, value) in expr.named_args:
            new_dict[key] = label_vars(value, path + delim + key, suffixes, delim)
        return scope.dict(**new_dict)

    # Default:
    inputs = [label_vars(v, path, suffixes, delim) for v in expr.inputs()]
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



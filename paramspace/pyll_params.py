"""
A factory for hyperopt's pyll parameter spaces.
"""
import scipy.stats.distributions
from pymbolic.mapper import WalkMapper
from pymbolic.mapper.evaluator import EvaluationMapper
from hyperopt import hp, pyll
import weakref


def label_key(obj):
    return (weakref.ref(obj), id(obj))


class PyllLabeler(WalkMapper):

    def __init__(self):
        super(PyllLabeler, self).__init__()
        self.labels = dict()

    def make_label(self, path, obj):
        key = label_key(obj)
        if not key in self.labels:
            self.labels[key] = path

    def map_call(self, expr, path=""):
        self.make_label(path, expr)
        super(PyllLabeler, self).map_call(expr, path=path)

    def map_call_with_kwargs(self, expr, path=""):
        for val in expr.parameters:
            self(val, path)
        for key, value in expr.kw_parameters.items():
            self(value, path = path + "." + key)


class PyllMapper(EvaluationMapper):

    def __init__(self, labels):
        super(PyllMapper, self).__init__()
        self.labels = labels

    def map_min(self, expr):
        args = [pyll.as_apply(self.rec(p)) for p in expr.children]
        return pyll.Apply("min", args, {})

    def map_max(self, expr):
        args = [pyll.as_apply(self.rec(p)) for p in expr.children]
        return pyll.Apply("max", args, {})

    def map_call(self, expr):
        label = self.labels[label_key(expr)]
        args = [self.rec(p) for p in expr.parameters]

        func_name = expr.function.name
        if func_name == "choice":
            return hp.choice(label, args)
        else:
            hp_func = getattr(hp, func_name)
            return hp_func(label, *args)

    def map_call_with_kwargs(self, expr):
        module_name, class_name = expr.function.name.rsplit(".", 1)

        kwargs = {key:self.rec(value)
                  for key, value in expr.kw_parameters.items()}
        kwargs["__name__"] = class_name
        kwargs["__module__"] = module_name
        return kwargs


def to_hp(expr):
    # First create the labels that will identify unique pyll nodes:
    labeler = PyllLabeler()
    labeler(expr)

    # Then convert the expression tree to pyll.
    # We can't create the labels and convert the tree in one step because
    # EvaluationMapper, from which PyllMapper inherits, does not support
    # arbitrary *args and **kwargs.
    mapper = PyllMapper(labeler.labels)
    return mapper(expr)


def class_space(cls):
    """
    Creates a parameter space for class `cls`.
    """
    space = dict()
    space["__module__"] = cls.__module__
    space["__name__"] = cls.__name__
    return space


def param_space(path, low=None, high=None, dist=None):
    """
    Creates a parameter space for a parameter defined by its bounds
    and/or distribution.
    """
    param = None

    if dist is not None:
        if isinstance(dist, scipy.stats.distributions.rv_frozen):
            # We have a scipy random variable.
            # Let's see if we can convert it to a pyll object:
            rv = dist
            if rv.dist.name == "uniform":
                dist = hp.uniform(path, *rv.args)
            elif rv.dist.name == "norm":
                dist = hp.normal(path, *rv.args)
            else:
                raise Exception("Unsupported distribution: %s" % rv.dist.name)
            # TODO implement lognormal

        if isinstance(dist, pyll.base.Apply):
            # We have a pyll object already.
            param = dist

            # Add our path prefix to the tree:
            pass # TODO FIXME

            # Did the user specify any hard bounds?
            if low is not None:
                param = pyll.Apply("max", (pyll.as_apply(low), param), {})
            if high is not None:
                param = pyll.Apply("min", (pyll.as_apply(high), param), {})

        else:
            raise Exception("Can't make sense of 'dist' argument: %s" % dist)

    elif not None in (low, high):
        # We only have the lower and upper bounds.
        # Assume a uniform distribution:
        param = hp.uniform(path, low, high)

    return param

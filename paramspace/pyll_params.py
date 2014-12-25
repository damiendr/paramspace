"""
A factory for hyperopt's pyll parameter spaces.
"""
import scipy.stats.distributions
from hyperopt import hp, pyll


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

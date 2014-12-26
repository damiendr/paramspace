"""
A factory for hyperopt's pyll parameter spaces.
"""
from pymbolic.mapper.evaluator import EvaluationMapper
from hyperopt import hp, pyll
from hyperopt.pyll.base import scope
import scipy.stats.distributions


class PyllMapper(EvaluationMapper):

    def map_variable(self, expr):
        for ctx in (hp, scope):
            try: return getattr(ctx, expr.name)
            except AttributeError: pass
        return expr.name

    def map_call_with_kwargs(self, expr):
        module_name, class_name = expr.function.name.rsplit(".", 1)
        kwargs = {key:self.rec(value)
                  for key, value in expr.kw_parameters.items()}
        kwargs["__name__"] = class_name
        kwargs["__module__"] = module_name
        return kwargs


def class_space(cls):
    """
    Creates a parameter space for class `cls`.
    """
    space = dict()
    space["__module__"] = cls.__module__
    space["__name__"] = cls.__name__
    return space


def param_space(low=None, high=None, dist=None):
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
                dist = scope.uniform(*rv.args)
            elif rv.dist.name == "norm":
                dist = scope.normal(*rv.args)
            else:
                raise Exception("Unsupported distribution: %s" % rv.dist.name)
            # TODO implement lognormal

        if isinstance(dist, pyll.base.Apply):
            # We have a pyll object already.
            param = dist

            # Did the user specify any hard bounds?
            if low is not None:
                param = scope.max(low, param)
            if high is not None:
                param = scope.min(high, param)

        else:
            raise Exception("Can't make sense of 'dist' argument: %s" % dist)

    elif not None in (low, high):
        # We only have the lower and upper bounds.
        # Assume a uniform distribution:
        param = scope.uniform(low, high)

    return param

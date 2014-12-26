from pymbolic.primitives import Variable, Call, CallWithKwargs
from pymbolic.mapper.evaluator import EvaluationMapper
from pymbolic.mapper import IdentityMapper
from collections import defaultdict


class Parameter(Call):
    pass


class Instance(CallWithKwargs):
    def default(self, name):
        return self.label_vars(path=name)

    def label_vars(self, path):
        return VariableLabeler()(self, path)


def choice(*options):
    return Parameter(Variable("choice"), (options,))


def randint(upper):
    return Parameter(Variable("randint"), (upper,))


def uniform(low, high):
    return Parameter(Variable("uniform"), (low, high))


def loguniform(low, high):
    return Parameter(Variable("loguniform"), (low, high))


def normal(mu, sigma):
    return Parameter(Variable("normal"), (mu, sigma))


def lognormal(mu, sigma):
    return Parameter(Variable("lognormal"), (mu, sigma))


def quniform(low, high, q):
    return Parameter(Variable("quniform"), (low, high, q))


def qloguniform(low, high, q):
    return Parameter(Variable("qloguniform"), (low, high, q))


def qnormal(mu, sigma, q):
    return Parameter(Variable("qnormal"), (mu, sigma, q))


def qlognormal(mu, sigma, q):
    return Parameter(Variable("qlognormal"), (mu, sigma, q))


def instance(cls, params):
    class_name = "%s.%s" % (cls.__module__, cls.__name__)
    return Instance(Variable(class_name), (), params)


def parameter(name, low=None, high=None, dist=None):
    if dist is not None:
        # We had a distribution already, but maybe the user specified hard
        # bounds in addition to it?
        if low is not None: dist = Call(Variable("max"), (low, dist))
        if high is not None: dist = Call(Variable("min"), (high, dist))
    else:
        # No distribution specified. Assume a uniform distribution and
        # use the low and high bounds:
        if low is None: low = 0
        if high is None:
            raise ValueError("high must be specified when dist is None.")
        dist = uniform(low, high)

    return dist


class VariableLabeler(IdentityMapper):

    def __init__(self):
        super(VariableLabeler, self).__init__()
        self.suffixes = defaultdict(lambda: 0)

    def map_call(self, expr, path=""):
        if not isinstance(expr, Parameter):
            return super(VariableLabeler, self).map_call(expr, path)

        # We have a parameter; let's create a label for it. The label is the
        # current path, plus a suffix when more than one variable share the
        # same path:
        idx = self.suffixes[path]
        self.suffixes[path] += 1
        if idx == 0: label = path
        else: label = "%s_%d" % (path, idx)
        
        # Rebuild the node with the extra label argument:
        function = self(expr.function, path)
        params_with_label = [Variable(label)] + [self(p, path) for p in expr.parameters]
        expr_with_label = type(expr)(function, params_with_label)
        return expr_with_label

    def map_call_with_kwargs(self, expr, path=""):
        function = self(expr.function, path)
        parameters = [self(p, path) for p in expr.parameters]
        kw_parameters = {key:self(value, path + "." + key)
                         for key, value in expr.kw_parameters.items()}
        return type(expr)(function, parameters, kw_parameters)


class PyllMapper(EvaluationMapper):


    def __init__(self):
        super(PyllMapper, self).__init__
        from hyperopt.pyll.base import scope
        from hyperopt import hp
        self.contexts = (hp, scope)


    def map_variable(self, expr):
        for ctx in self.contexts:
            try: return getattr(ctx, expr.name)
            except AttributeError: pass
        return expr.name

    def map_call_with_kwargs(self, expr):
        kwargs = {key:self.rec(value)
                  for key, value in expr.kw_parameters.items()}
        kwargs["__class__"] = expr.function.name
        return kwargs


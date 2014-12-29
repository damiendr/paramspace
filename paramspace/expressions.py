from pymbolic.primitives import Variable, Call, CallWithKwargs
from pymbolic.mapper.evaluator import EvaluationMapper
from pymbolic.mapper import IdentityMapper
from collections import defaultdict


class Parameter(Call):
    pass


class Instance(CallWithKwargs):
    pass


def label_vars(root, space):
    return VariableLabeler()(space, root)


def to_hp(root, space):
    from hyperopt.pyll.base import Apply
    if isinstance(space, Apply): return space

    if root is not None:
        space = label_vars(root, space)
    return PyllMapper()(space)


def choice(*options):
    return Parameter(Variable("choice"), (options,))


def pchoice(*options):
    return Parameter(Variable("pchoice"), (options,))


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
        # We have a distribution already, but maybe the user specified
        # hard bounds in addition to it?
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
    """
    Adds labels that identify independant random variables.
    """
    def __init__(self):
        super(VariableLabeler, self).__init__()
        self.suffixes = defaultdict(lambda: 0)

    def map_call(self, expr, path=""):
        if not isinstance(expr, Parameter):
            return super(VariableLabeler, self).map_call(expr, path)

        # We have a parameter, let's create a label for it.

        # The label is the current path, plus a suffix when more than
        # one variable share the same path:
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
        prefix = path + "." if path else ""
        kw_parameters = {key:self(value, prefix + key)
                         for key, value in expr.kw_parameters.items()}
        return type(expr)(function, parameters, kw_parameters)


class PyllMapper(EvaluationMapper):
    """
    Converts a Pymbolic expression to the equivalent Pyll expression.
    """
    def __init__(self):
        super(PyllMapper, self).__init__()

        # Setup the contexts we will use for resolving function names:
        from hyperopt.pyll.base import scope
        from hyperopt import hp
        self.contexts = (hp, scope) # in order of precedence

    def map_variable(self, expr):
        # Try to resolve the variable as a hyperopt function:
        for ctx in self.contexts:
            try: return getattr(ctx, expr.name)
            except AttributeError: pass

        # Fallback: assume it's a string literal:
        return expr.name

    def map_call_with_kwargs(self, expr):
        if isinstance(expr, Instance):
            # It's a class instance, convert it to a dict with a
            # __class__ member:
            kwargs = {key:self.rec(value)
                      for key, value in expr.kw_parameters.items()}
            kwargs["__class__"] = expr.function.name
            from hyperopt.pyll.base import as_apply
            return as_apply(kwargs)

        return super(PyllMapper, self).map_call_with_kwargs(expr)


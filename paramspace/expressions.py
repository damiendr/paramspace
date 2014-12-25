from pymbolic.primitives import Variable, Lookup, Call, CallWithKwargs, Min, Max


def choice(*options):
    return Call(Variable("choice"), options)


def randint(upper):
    return Call(Variable("randint"), (upper,))


def uniform(low, high):
    return Call(Variable("uniform"), (low, high))


def instance(cls, params):
    class_name = "%s.%s" % (cls.__module__, cls.__name__)
    return CallWithKwargs(Variable(class_name), (), params)


def parameter(name, low=None, high=None, dist=None):
    if dist is not None:
        # We had a distribution already, but maybe the user specified hard
        # bounds in addition to it?
        if low is not None: dist = Max(low, dist)
        if high is not None: dist = Min(high, dist)
    else:
        # No distribution specified. Assume a uniform distribution and
        # use the low and high bounds:
        if low is None: low = 0
        if high is None:
            raise ValueError("high must be specified when dist is None.")
        dist = uniform(low, high)

    return dist



from paramspace.expressions import label_vars, to_hp


class sample(object):
    """
    Helper class for use with `callipy` parameters:

        %param foo sample(foo_space)

    This will draw a random sample from `foo_space` as a default value.
    """
    def __init__(self, space):
        self.space = space

    def default(self, param_name):
        return load_param(param_name, self.space)


def load_param(name, space, scope=None):
    """
    Loads a parameter specified by its `name` from the given `scope`.

    When no such parameter can be found in `scope`, a random sample will be
    drawn from the given parameter `space`.
    
    Finally, the parameter value is decoded using `load_model(value)`.
    """
    if scope and name in scope:
        # Use the given value:
        value = scope[name]
    else:
        # Draw a random sample from the parameter space using pyll:
        from hyperopt.pyll import stochastic
        hp_space = to_hp(label_vars(space, name))
        value = stochastic.sample(hp_space)

    # Decode the value into an object tree:
    return load_model(value)


def load_model(obj):
    """
    Decodes a point in parameter space into an object tree, instanciating
    classes where applicable.
    """
    if isinstance(obj, dict):
        # First decode the contents:
        kwargs = {k:load_model(v) for k,v in obj.items()}

        try: # Can we decode the dict as a class?
            import importlib
            module_name, class_name = kwargs.pop("__class__").rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

        except KeyError: # Nope, it was just a normal dict
            return kwargs
        
        else: # We have a class! Let's instanciate it:
            return cls(**kwargs)

    elif isinstance(obj, (list, tuple)):
        # Decode the contents and return a container of the same type:
        return type(obj)(load_model(x) for x in obj)

    # Default: return obj as it is.
    return obj


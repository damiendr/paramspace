from traits.api import HasTraits
import traits.trait_types
import logging


def build_trait_space(sp, name, trait_type):
    """
    Builds a parameter space for a trait of type `trait_type`.
    """
    if isinstance(trait_type, traits.trait_types.Instance):
        cls = trait_type.klass
        # When the instance type is a subclass of HasTraits,
        # we can build a subspace for its traits, recursively:
        if issubclass(cls, HasTraits):
            return build_class_space(cls, sp)

    if isinstance(trait_type, (traits.trait_types.Enum, 
                                traits.trait_types.List,
                                traits.trait_types.Tuple,
                                traits.trait_types.Dict)):
        # TODO look into container traits
        logging.warn("Looking for parameters inside container traits"
                     "is not yet implemented")

    # The user may specify a distribution in the trait metadata:
    dist = trait_type._metadata.get("dist", None)

    # Do we have any information about bounds?
    low = high = None
    if isinstance(trait_type, traits.trait_types.Range):
        low = trait_type._low
        high = trait_type._high

    return sp.param_space(name, low, high, dist)


def build_class_space(cls, sp, **kwargs):
    """
    Builds a parameter space for class `cls` inferred from the class' traits,
    except where overriden by `kwargs`.
    """
    # Does this class define any traits?
    try: traits = cls.__class_traits__
    except NameError: traits = {}

    # Store the class in the parameter space, so that we can deserialise a
    # point in that parameter space into the corresponding object tree:
    space = sp.class_space(cls)

    for name, ctrait in traits.items():
        # We're only interested in user-defined traits, but Trait also stores
        # events & such as traits. Skip these:
        if ctrait.type != "trait": continue

        # Did the user provide a parameter space for this trait?
        # If not, build a default parameter space.
        try: tspace = kwargs.pop(name)
        except KeyError: tspace = build_trait_space(sp, name, ctrait.trait_type)

        # Do we have a valid parameter space?
        # If not, let's just ignore this trait.
        if tspace is not None: space[name] = tspace

    if kwargs: # We should have popped all the values in kwargs by now:
        raise NameError("%s has no traits named %s." % (cls, kwargs.keys()))

    return space


class ModelTraits(HasTraits):
    """
    A subclass of `HasTraits` that provides a convenient `space()`
    class method returning the parameter space for that class.
    """
    @classmethod
    def space(cls, **kwargs):
        """
        Returns the parameter space for `cls`. Sub-spaces for the class' traits
        will be taken from `kwargs`, when supplied, or inferred from the trait
        type. Traits for which no sub-space can be inferred will be ignored.
        """
        from experiments.params import pyll_params
        return build_class_space(cls, pyll_params, **kwargs)


def load(obj):
    """
    Decodes a point in parameter space into an object tree,
    instanciating classes where applicable.
    """
    if isinstance(obj, dict):
        # First decode the contents:
        kwargs = {k:load(v) for k,v in obj.items()}

        try: # Can we decode the dict as a class?
            import importlib
            module = importlib.import_module(kwargs.pop("__module__"))
            cls = getattr(module, kwargs.pop("__name__"))
        except KeyError: # Nope, it was just a normal dict
            return kwargs
        else: # We have a class! Let's instanciate it:
            return cls(**kwargs)

    elif isinstance(obj, (list, tuple)):
        # Decode the contents and return a container of the same type:
        return type(obj)(load(x) for x in obj)

    # Default: return obj as it is.
    return obj

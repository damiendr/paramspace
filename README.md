
# Paramspace

A bridge between declarative models, parameter spaces and parameter optimisation
tools.

Development in alpha stage.

## What is it?

Packages such as Enthought's
[Traits](http://code.enthought.com/projects/traits/) make it easy to define
models and their parameters in a declarative way.

On the other hand, one can use tools such as
[Hyperopt](http://hyperopt.github.io/hyperopt/) to find optimal parameters in a
given search space.

Paramspace creates a bridge between the two: if you have a model definition, you
can extract a parameter space from it; and if you have a sample in that
parameter space, you can decode it back into an instance of your model
configured with the appropriate parameters.

## Install

First install the dependencies:
- [hyperopt](http://hyperopt.github.io/hyperopt) >= 0.0.2 (from pip)
- [traits](http://code.enthought.com/projects/traits/) >= 4.5.0 (from pip)

Then clone this repository and run `python setup.py install`.

## Usage

First, let us define a sample model with `traits`. A model can be any subclass
of `HasTraits`, but paramspace also provides a special subclass, `ModelTraits`,
which comes with some useful methods.


```python
from paramspace.traits_model import ModelTraits
from paramspace import choice, uniform, load_param, load_model, label_vars
from traits.api import Float, Any
```


```python
class NeuronModel(ModelTraits):
    lr = Float(dist=uniform(1e-5, 1e-3))
    learn = Any()
    
    # here goes the model logic
    # ...

class StandardHebb(ModelTraits):
    
    def dW(self, x, y):
        return np.dot(x, y)

    
class DecayHebb(ModelTraits):
    decay = Float()

    def dW(self, x, y):
        return np.dot(x, y) - self.decay

```

Now let's build a parameter space. Note how the parameter `lr` was automatically
extracted from the model definition, while `learn` and `decay` are given
explicitely. This is because both of these lack the metadata to infer the
corresponding parameter space.


```python
search_space = NeuronModel.space(
    learn = choice(
        StandardHebb.space(),
        DecayHebb.space(
            decay = 0.1 * uniform(0, 1)
        )
    )
)

print search_space
```

    0 dict
    1  __class__ =
    2   Literal{__main__.NeuronModel}
    3  learn =
    4   switch
    5     randint
    6       Literal{2}
    7     dict
    8      __class__ =
    9       Literal{__main__.StandardHebb}
    10     dict
    11      __class__ =
    12       Literal{__main__.DecayHebb}
    13      decay =
    14       mul
    15         Literal{0.1}
    16         uniform
    17           Literal{0}
    18           Literal{1}
    19  lr =
    20   uniform
    21     Literal{1e-05}
    22     Literal{0.001}


We can now draw a random sample from that space:


```python
neuron = load_param("neuron", search_space)
print neuron
```

    <__main__.NeuronModel object at 0x111b2fe90>


Let's inspect the resulting object:


```python
import yaml
print yaml.dump(neuron)
```

    !!python/object:__main__.NeuronModel
    __traits_version__: 4.5.0
    learn: !!python/object:__main__.DecayHebb {__traits_version__: 4.5.0, decay: 0.0018778939663813832}
    lr: 8.507942215338842e-05
    


We can also convert the parameter space to a form that `hyperopt` can sample:


```python
hpspace = label_vars(search_space, path="neuron")
print hpspace
```

    0 dict
    1  __class__ =
    2   Literal{__main__.NeuronModel}
    3  learn =
    4   switch
    5     hyperopt_param
    6       Literal{neuron.learn}
    7       randint
    8         Literal{2}
    9     dict
    10      __class__ =
    11       Literal{__main__.StandardHebb}
    12     dict
    13      __class__ =
    14       Literal{__main__.DecayHebb}
    15      decay =
    16       mul
    17         Literal{0.1}
    18         float
    19           hyperopt_param
    20             Literal{neuron.learn.decay}
    21             uniform
    22               Literal{0}
    23               Literal{1}
    24  lr =
    25   float
    26     hyperopt_param
    27       Literal{neuron.lr}
    28       uniform
    29         Literal{1e-05}
    30         Literal{0.001}



```python
from hyperopt.pyll.stochastic import sample
s = sample(hpspace)
print s
```

    {'lr': 0.000919081401258073, '__class__': '__main__.NeuronModel', 'learn': {'__class__': '__main__.DecayHebb', 'decay': 0.023754228985201298}}


... and convert a sample from hyperopt back into a model instance:


```python
neuron = load_model(s)
print neuron
print yaml.dump(neuron)
```

    <__main__.NeuronModel object at 0x111b411d0>
    !!python/object:__main__.NeuronModel
    __traits_version__: 4.5.0
    learn: !!python/object:__main__.DecayHebb {__traits_version__: 4.5.0, decay: 0.023754228985201298}
    lr: 0.000919081401258073
    


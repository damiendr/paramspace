# Paramspace

A bridge between declarative models, parameter spaces and parameter optimisation tools.

## What is it?

Packages such as Enthought's [Traits](http://code.enthought.com/projects/traits/) make it easy to define models and their parameters in a declarative way.

On the other hand, one can use tools such as [Hyperopt](http://hyperopt.github.io/hyperopt/) to find optimal parameters in a given search space.

Paramspace creates a bridge between the two: if you have a model definition, you can extract a parameter space from it; and if you have a sample in that parameter space, you can decode it back into an instance of your model configured with the appropriate parameters.

## Install

First install the dependencies:
- [hyperopt](hyperopt.github.io/hyperopt) >= 0.0.2 (from pip)
- [traits](http://code.enthought.com/projects/traits/) >= 4.5.0 (from pip)

Then clone this repository and run `python setup.py install`.

## Usage

First, let us define a sample model with `traits`. A model can be any subclass of `HasTraits`, but paramspace also provides a special subclass, `ModelTraits`, which comes with some useful methods.

```{.python .input n=22}
from paramspace.traits_model import ModelTraits
from paramspace import choice, uniform, load_param, load_model, label_vars
from traits.api import Float, Any
```

```{.python .input n=23}
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

Now let's build a parameter space. Note how the parameter `lr` was automatically extracted from the model definition, while `learn` and `decay` are given explicitely. This is because both of these lack the metadata to infer the corresponding parameter space.

```{.python .input n=24}
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

```{.json .output n=24}
[
 {
  "output_type": "stream",
  "stream": "stdout",
  "text": "0 dict\n1  __class__ =\n2   Literal{__main__.NeuronModel}\n3  learn =\n4   switch\n5     randint\n6       Literal{2}\n7     dict\n8      __class__ =\n9       Literal{__main__.StandardHebb}\n10     dict\n11      __class__ =\n12       Literal{__main__.DecayHebb}\n13      decay =\n14       mul\n15         Literal{0.1}\n16         uniform\n17           Literal{0}\n18           Literal{1}\n19  lr =\n20   uniform\n21     Literal{1e-05}\n22     Literal{0.001}\n"
 }
]
```

We can now draw a random sample from that space:

```{.python .input n=14}
neuron = load_param("neuron", search_space)
print neuron
```

```{.json .output n=14}
[
 {
  "output_type": "stream",
  "stream": "stdout",
  "text": "<__main__.NeuronModel object at 0x111b2fe90>\n"
 }
]
```

Let's inspect the resulting object:

```{.python .input n=15}
import yaml
print yaml.dump(neuron)
```

```{.json .output n=15}
[
 {
  "output_type": "stream",
  "stream": "stdout",
  "text": "!!python/object:__main__.NeuronModel\n__traits_version__: 4.5.0\nlearn: !!python/object:__main__.DecayHebb {__traits_version__: 4.5.0, decay: 0.0018778939663813832}\nlr: 8.507942215338842e-05\n\n"
 }
]
```

We can also convert the parameter space to a form that `hyperopt` can sample:

```{.python .input n=20}
hpspace = label_vars(search_space, path="neuron")
print hpspace
```

```{.json .output n=20}
[
 {
  "output_type": "stream",
  "stream": "stdout",
  "text": "0 dict\n1  __class__ =\n2   Literal{__main__.NeuronModel}\n3  learn =\n4   switch\n5     hyperopt_param\n6       Literal{neuron.learn}\n7       randint\n8         Literal{2}\n9     dict\n10      __class__ =\n11       Literal{__main__.StandardHebb}\n12     dict\n13      __class__ =\n14       Literal{__main__.DecayHebb}\n15      decay =\n16       mul\n17         Literal{0.1}\n18         float\n19           hyperopt_param\n20             Literal{neuron.learn.decay}\n21             uniform\n22               Literal{0}\n23               Literal{1}\n24  lr =\n25   float\n26     hyperopt_param\n27       Literal{neuron.lr}\n28       uniform\n29         Literal{1e-05}\n30         Literal{0.001}\n"
 }
]
```

```{.python .input n=21}
from hyperopt.pyll.stochastic import sample
s = sample(hpspace)
print s
```

```{.json .output n=21}
[
 {
  "output_type": "stream",
  "stream": "stdout",
  "text": "{'lr': 0.000919081401258073, '__class__': '__main__.NeuronModel', 'learn': {'__class__': '__main__.DecayHebb', 'decay': 0.023754228985201298}}\n"
 }
]
```

... and convert a sample from hyperopt back into a model instance:

```{.python .input n=26}
neuron = load_model(s)
print neuron
print yaml.dump(neuron)
```

```{.json .output n=26}
[
 {
  "output_type": "stream",
  "stream": "stdout",
  "text": "<__main__.NeuronModel object at 0x111b411d0>\n!!python/object:__main__.NeuronModel\n__traits_version__: 4.5.0\nlearn: !!python/object:__main__.DecayHebb {__traits_version__: 4.5.0, decay: 0.023754228985201298}\nlr: 0.000919081401258073\n\n"
 }
]
```

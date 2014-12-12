

from paramspace.traits_model import ModelTraits, load
from hyperopt import hp, pyll
from traits.api import Float, Range, Instance


class TestModel(ModelTraits):
	x = Float(0.0)
	y = Float(0.0, dist=hp.uniform("y", 0, 1))
	z = Range(low=0.0, dist=hp.normal("z", 0, 1))


class TestModel2(ModelTraits):
	a = Instance(TestModel)


def test_simple_traits():
	s = TestModel.space()
	assert set(s.keys()) == {'y', 'z', '__name__', '__module__'}

	s = TestModel.space(x = hp.randint('x', 0, 5))
	assert set(s.keys()) == {'x', 'y', 'z', '__name__', '__module__'}


def test_sample_load():
	s = TestModel.space()
	p = pyll.stochastic.sample(s)
	t = load(p)
	assert isinstance(t, TestModel)


def test_load():
	s = TestModel.space()
	p = dict(s)
	p['y'] = 1.0
	p['z'] = 0.5
	t = load(p)

	assert t.y == p['y']
	assert t.z == p['z']


def test_nested_traits():
	s2 = TestModel2.space()
	assert set(s2.keys()) == {'a', '__name__', '__module__'}

	s = {
		'__name__': TestModel2.__name__,
		'__module__': __name__,
		'a': {
			'__name__': TestModel.__name__,
			'__module__': __name__,
			'y': 1.0,
			'z': 5.0,
		}
	}
	t = load(s)
	assert isinstance(t, TestModel2)
	assert isinstance(t.a, TestModel)
	assert t.a.y == 1.0
	assert t.a.z == 5.0
	

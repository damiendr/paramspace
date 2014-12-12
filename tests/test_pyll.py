from paramspace.pyll_params import param_space
from hyperopt import hp, pyll
import scipy.stats.distributions


def test_dist_uniform():
    p1 = param_space("x", dist=scipy.stats.distributions.uniform(2,3))
    p2 = param_space("x", dist=hp.uniform("x", 2, 3))
    assert str(p1) == str(p2)

def test_dist_norm():
    p1 = param_space("x", dist=scipy.stats.distributions.norm(2,3))
    p2 = param_space("x", dist=hp.normal("x", 2, 3))
    assert str(p1) == str(p2)

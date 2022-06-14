# test the Suggest-Evaluate-Register Paradigm

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

# Let's start by defining our function, bounds, and instanciating an optimization object.
def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

optimizer = BayesianOptimization(
    f=None,
    pbounds={'x': (-2, 2), 'y': (-3, 3)},
    verbose=2,
    random_state=1,
)

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

for _ in range(5):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)
    
    print(target, next_point)
    
print(optimizer.max)
import rbfopt
import numpy as np

settings = rbfopt.RbfoptSettings(minlp_solver_path='~/bonmin-linux64/bonmin', nlp_solver_path='~/ipopt-linux64/ipopt', max_evaluations=50)


def obj_funct(x):
  return x[0]*x[1] - x[2]

bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                               np.array(['R', 'R', 'R']), obj_funct)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()

print(val, x, itercount, evalcount, fast_evalcount)
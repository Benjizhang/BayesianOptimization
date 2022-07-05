util = UtilityFunction(kind="ei", 
                        kappa = 2, 
                        xi=0.5,
                        kappa_decay=1,
                        kappa_decay_delay=0)
						
** summary of exp **
distribution, length_sacle, length_scale_bounds, max_iteration: results
**
exp1
small, 1, fixed, 30: bad
exp2
small, 1, -, 30: bad
exp3
small, 8, -, 30: bad
exp4
small, 8, -, 30: bad
exp5 
large, 8, -, 30: bad
exp6
large, 8, -, 50: bad
exp7
large, 1, -, 50: bad

exp8
large, 0.04, fixed, 20, good
exp9
large, 0.04, deflt, 30: good
exp10
small, 0.04, deflt, 30: good
exp11
small, 0.04, fixed, 30: bad
exp12
small, 0.04, fixed, 60: bad 
exp13
large, 0.04, fixed, 60: good 
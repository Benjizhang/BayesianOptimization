util = UtilityFunction(kind="ei", 
					kappa = 2, 
					xi=0.5,
					kappa_decay=1,
					kappa_decay_delay=0)
** summary of exp **
distribution, length_sacle, length_scale_bounds, max_iteration: results
**
exp1
small, 0.04, fixed, 30: bad
exp2
small, 0.04, default, 30: bad
exp3
small, 0.04, [.01,.2], 30: bad
exp4 (smooth_f)
small, 0.04, fixed, 40: good
exp5 (smooth_f)
small, 0.04, fixed, 50: good
exp6 (smooth_f)
small, 0.04, fixed, 30: good
exp7 (smooth_f)
small, 0.04, [.01,.2], 60: bad 
exp8 (smooth_f)
small, 0.04, [.01,.2], 40: bad
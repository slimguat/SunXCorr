
# Function definition
import numpy as np
from numba import jit

def model_function(x,arg_0,arg_1,arg_2,arg_3):
	l=4
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_210562 = (x > 778.0017318720002) & (x < 782.8775818720001);sub_x_210562 = x[mask_array_210562];sum[mask_array_210562] += arg_3
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3):
	l=4
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_76745374 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_47854432 = (x > 778.0017318720002) & (x < 782.8775818720001);sub_x_47854432 = x[mask_array_47854432]
	
	jac[:,0]= exp_76745374
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_76745374
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_76745374
	jac[mask_array_47854432,3]= 1
	
	
	return jac

# Function definition
import numpy as np
from numba import jit

def model_function(x,arg_0,arg_1,arg_2,arg_3):
	l=4
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_50892959 = (x > 778.0017318720002) & (x < 782.8775818720001);sub_x_50892959 = x[mask_array_50892959];sum[mask_array_50892959] += arg_3
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3):
	l=4
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_26156410 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	mask_array_53356936 = (x > 778.0017318720002) & (x < 782.8775818720001);sub_x_53356936 = x[mask_array_53356936]
	
	jac[:,0]= exp_26156410
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_26156410
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_26156410
	jac[mask_array_53356936,3]= 1
	
	
	return jac
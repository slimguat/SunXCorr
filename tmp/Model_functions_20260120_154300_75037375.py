
# Function definition
import numpy as np
from numba import jit

def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9):
	l=10
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	sum+=arg_3*np.exp(-(x-arg_4)**2/(2*arg_5**2))
	sum+=arg_6*np.exp(-(x-arg_7)**2/(2*arg_8**2))
	mask_array_47128230 = (x > 782.682547872) & (x < 790.4839078720001);sub_x_47128230 = x[mask_array_47128230];sum[mask_array_47128230] += arg_9
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9):
	l=10
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_35982579 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_54128394 = np.exp(-(x-arg_4)**2/(2*arg_5**2))
	exp_57148874 = np.exp(-(x-arg_7)**2/(2*arg_8**2))
	mask_array_11751462 = (x > 782.682547872) & (x < 790.4839078720001);sub_x_11751462 = x[mask_array_11751462]
	
	jac[:,0]= exp_35982579
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_35982579
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_35982579
	jac[:,3]= exp_54128394
	jac[:,4]= arg_3*(x-arg_4)   /(arg_5**2) * exp_54128394
	jac[:,5]= arg_3*(x-arg_4)**2/(arg_5**3) * exp_54128394
	jac[:,6]= exp_57148874
	jac[:,7]= arg_6*(x-arg_7)   /(arg_8**2) * exp_57148874
	jac[:,8]= arg_6*(x-arg_7)**2/(arg_8**3) * exp_57148874
	jac[mask_array_11751462,9]= 1
	
	
	return jac
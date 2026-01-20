
# Function definition
import numpy as np
from numba import jit

def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-arg_1)**2/(2*arg_2**2))
	sum+=arg_3*np.exp(-(x-(arg_1+2.0699999999999363))**2/(2*(arg_2+0.0)**2))
	sum+=arg_4*np.exp(-(x-arg_5)**2/(2*arg_6**2))
	sum+=arg_7*np.exp(-(x-arg_8)**2/(2*arg_9**2))
	sum+=arg_10*np.exp(-(x-arg_11)**2/(2*arg_12**2))
	sum+=arg_13*np.exp(-(x-(arg_1+8.92999999999995))**2/(2*(arg_2+0.0)**2))
	mask_array_84221874 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_84221874 = x[mask_array_84221874];sum[mask_array_84221874] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_29309261 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_91136820 = np.exp(-(x-(arg_1+2.0699999999999363))**2/(2*(arg_2+0.0)**2))
	exp_51077804 = np.exp(-(x-arg_5)**2/(2*arg_6**2))
	exp_17742934 = np.exp(-(x-arg_8)**2/(2*arg_9**2))
	exp_14959929 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	exp_19243337 = np.exp(-(x-(arg_1+8.92999999999995))**2/(2*(arg_2+0.0)**2))
	mask_array_53377702 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_53377702 = x[mask_array_53377702]
	
	jac[:,0]= exp_29309261
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_29309261 + arg_3*(x-(arg_1+2.0699999999999363))   /((arg_2+0.0)**2) * exp_91136820 + arg_13*(x-(arg_1+8.92999999999995))   /((arg_2+0.0)**2) * exp_19243337
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_29309261 + arg_3*(x-(arg_1+2.0699999999999363))**2/((arg_2+0.0)**3) * exp_91136820 + arg_13*(x-(arg_1+8.92999999999995))**2/((arg_2+0.0)**3) * exp_19243337
	jac[:,3]= exp_91136820
	jac[:,4]= exp_51077804
	jac[:,5]= arg_4*(x-arg_5)   /(arg_6**2) * exp_51077804
	jac[:,6]= arg_4*(x-arg_5)**2/(arg_6**3) * exp_51077804
	jac[:,7]= exp_17742934
	jac[:,8]= arg_7*(x-arg_8)   /(arg_9**2) * exp_17742934
	jac[:,9]= arg_7*(x-arg_8)**2/(arg_9**3) * exp_17742934
	jac[:,10]= exp_14959929
	jac[:,11]= arg_10*(x-arg_11)   /(arg_12**2) * exp_14959929
	jac[:,12]= arg_10*(x-arg_11)**2/(arg_12**3) * exp_14959929
	jac[:,13]= exp_19243337
	jac[mask_array_53377702,14]= 1
	
	
	return jac
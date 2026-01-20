
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
	mask_array_56532884 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_56532884 = x[mask_array_56532884];sum[mask_array_56532884] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_53820136 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_389595 = np.exp(-(x-(arg_1+2.0699999999999363))**2/(2*(arg_2+0.0)**2))
	exp_74905446 = np.exp(-(x-arg_5)**2/(2*arg_6**2))
	exp_40025010 = np.exp(-(x-arg_8)**2/(2*arg_9**2))
	exp_43616657 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	exp_8700265 = np.exp(-(x-(arg_1+8.92999999999995))**2/(2*(arg_2+0.0)**2))
	mask_array_98548366 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_98548366 = x[mask_array_98548366]
	
	jac[:,0]= exp_53820136
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_53820136 + arg_3*(x-(arg_1+2.0699999999999363))   /((arg_2+0.0)**2) * exp_389595 + arg_13*(x-(arg_1+8.92999999999995))   /((arg_2+0.0)**2) * exp_8700265
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_53820136 + arg_3*(x-(arg_1+2.0699999999999363))**2/((arg_2+0.0)**3) * exp_389595 + arg_13*(x-(arg_1+8.92999999999995))**2/((arg_2+0.0)**3) * exp_8700265
	jac[:,3]= exp_389595
	jac[:,4]= exp_74905446
	jac[:,5]= arg_4*(x-arg_5)   /(arg_6**2) * exp_74905446
	jac[:,6]= arg_4*(x-arg_5)**2/(arg_6**3) * exp_74905446
	jac[:,7]= exp_40025010
	jac[:,8]= arg_7*(x-arg_8)   /(arg_9**2) * exp_40025010
	jac[:,9]= arg_7*(x-arg_8)**2/(arg_9**3) * exp_40025010
	jac[:,10]= exp_43616657
	jac[:,11]= arg_10*(x-arg_11)   /(arg_12**2) * exp_43616657
	jac[:,12]= arg_10*(x-arg_11)**2/(arg_12**3) * exp_43616657
	jac[:,13]= exp_8700265
	jac[mask_array_98548366,14]= 1
	
	
	return jac
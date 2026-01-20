
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
	mask_array_42038693 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_42038693 = x[mask_array_42038693];sum[mask_array_42038693] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_88853700 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_23906809 = np.exp(-(x-(arg_1+2.0699999999999363))**2/(2*(arg_2+0.0)**2))
	exp_63383245 = np.exp(-(x-arg_5)**2/(2*arg_6**2))
	exp_10296245 = np.exp(-(x-arg_8)**2/(2*arg_9**2))
	exp_84562432 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	exp_78931454 = np.exp(-(x-(arg_1+8.92999999999995))**2/(2*(arg_2+0.0)**2))
	mask_array_19774319 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_19774319 = x[mask_array_19774319]
	
	jac[:,0]= exp_88853700
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_88853700 + arg_3*(x-(arg_1+2.0699999999999363))   /((arg_2+0.0)**2) * exp_23906809 + arg_13*(x-(arg_1+8.92999999999995))   /((arg_2+0.0)**2) * exp_78931454
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_88853700 + arg_3*(x-(arg_1+2.0699999999999363))**2/((arg_2+0.0)**3) * exp_23906809 + arg_13*(x-(arg_1+8.92999999999995))**2/((arg_2+0.0)**3) * exp_78931454
	jac[:,3]= exp_23906809
	jac[:,4]= exp_63383245
	jac[:,5]= arg_4*(x-arg_5)   /(arg_6**2) * exp_63383245
	jac[:,6]= arg_4*(x-arg_5)**2/(arg_6**3) * exp_63383245
	jac[:,7]= exp_10296245
	jac[:,8]= arg_7*(x-arg_8)   /(arg_9**2) * exp_10296245
	jac[:,9]= arg_7*(x-arg_8)**2/(arg_9**3) * exp_10296245
	jac[:,10]= exp_84562432
	jac[:,11]= arg_10*(x-arg_11)   /(arg_12**2) * exp_84562432
	jac[:,12]= arg_10*(x-arg_11)**2/(arg_12**3) * exp_84562432
	jac[:,13]= exp_78931454
	jac[mask_array_19774319,14]= 1
	
	
	return jac
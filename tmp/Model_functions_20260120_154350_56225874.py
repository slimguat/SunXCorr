
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
	mask_array_17195918 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_17195918 = x[mask_array_17195918];sum[mask_array_17195918] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_69692804 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_74555347 = np.exp(-(x-(arg_1+2.0699999999999363))**2/(2*(arg_2+0.0)**2))
	exp_32476403 = np.exp(-(x-arg_5)**2/(2*arg_6**2))
	exp_42323005 = np.exp(-(x-arg_8)**2/(2*arg_9**2))
	exp_2821836 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	exp_79027887 = np.exp(-(x-(arg_1+8.92999999999995))**2/(2*(arg_2+0.0)**2))
	mask_array_97929810 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_97929810 = x[mask_array_97929810]
	
	jac[:,0]= exp_69692804
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_69692804 + arg_3*(x-(arg_1+2.0699999999999363))   /((arg_2+0.0)**2) * exp_74555347 + arg_13*(x-(arg_1+8.92999999999995))   /((arg_2+0.0)**2) * exp_79027887
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_69692804 + arg_3*(x-(arg_1+2.0699999999999363))**2/((arg_2+0.0)**3) * exp_74555347 + arg_13*(x-(arg_1+8.92999999999995))**2/((arg_2+0.0)**3) * exp_79027887
	jac[:,3]= exp_74555347
	jac[:,4]= exp_32476403
	jac[:,5]= arg_4*(x-arg_5)   /(arg_6**2) * exp_32476403
	jac[:,6]= arg_4*(x-arg_5)**2/(arg_6**3) * exp_32476403
	jac[:,7]= exp_42323005
	jac[:,8]= arg_7*(x-arg_8)   /(arg_9**2) * exp_42323005
	jac[:,9]= arg_7*(x-arg_8)**2/(arg_9**3) * exp_42323005
	jac[:,10]= exp_2821836
	jac[:,11]= arg_10*(x-arg_11)   /(arg_12**2) * exp_2821836
	jac[:,12]= arg_10*(x-arg_11)**2/(arg_12**3) * exp_2821836
	jac[:,13]= exp_79027887
	jac[mask_array_97929810,14]= 1
	
	
	return jac
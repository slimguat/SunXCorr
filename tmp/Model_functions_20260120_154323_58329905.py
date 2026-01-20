
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
	mask_array_41934086 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_41934086 = x[mask_array_41934086];sum[mask_array_41934086] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_92018232 = np.exp(-(x-arg_1)**2/(2*arg_2**2))
	exp_531554 = np.exp(-(x-(arg_1+2.0699999999999363))**2/(2*(arg_2+0.0)**2))
	exp_19030668 = np.exp(-(x-arg_5)**2/(2*arg_6**2))
	exp_99150508 = np.exp(-(x-arg_8)**2/(2*arg_9**2))
	exp_43071736 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	exp_15664380 = np.exp(-(x-(arg_1+8.92999999999995))**2/(2*(arg_2+0.0)**2))
	mask_array_90874853 = (x > 984.6250261890001) & (x < 995.9801661889999);sub_x_90874853 = x[mask_array_90874853]
	
	jac[:,0]= exp_92018232
	jac[:,1]= arg_0*(x-arg_1)   /(arg_2**2) * exp_92018232 + arg_3*(x-(arg_1+2.0699999999999363))   /((arg_2+0.0)**2) * exp_531554 + arg_13*(x-(arg_1+8.92999999999995))   /((arg_2+0.0)**2) * exp_15664380
	jac[:,2]= arg_0*(x-arg_1)**2/(arg_2**3) * exp_92018232 + arg_3*(x-(arg_1+2.0699999999999363))**2/((arg_2+0.0)**3) * exp_531554 + arg_13*(x-(arg_1+8.92999999999995))**2/((arg_2+0.0)**3) * exp_15664380
	jac[:,3]= exp_531554
	jac[:,4]= exp_19030668
	jac[:,5]= arg_4*(x-arg_5)   /(arg_6**2) * exp_19030668
	jac[:,6]= arg_4*(x-arg_5)**2/(arg_6**3) * exp_19030668
	jac[:,7]= exp_99150508
	jac[:,8]= arg_7*(x-arg_8)   /(arg_9**2) * exp_99150508
	jac[:,9]= arg_7*(x-arg_8)**2/(arg_9**3) * exp_99150508
	jac[:,10]= exp_43071736
	jac[:,11]= arg_10*(x-arg_11)   /(arg_12**2) * exp_43071736
	jac[:,12]= arg_10*(x-arg_11)**2/(arg_12**3) * exp_43071736
	jac[:,13]= exp_15664380
	jac[mask_array_90874853,14]= 1
	
	
	return jac
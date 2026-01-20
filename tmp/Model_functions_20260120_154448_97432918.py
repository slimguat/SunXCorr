
# Function definition
import numpy as np
from numba import jit

def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11):
	l=12
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-(arg_4+-4.024000000000001))**2/(2*(arg_5+0.0)**2))
	sum+=arg_1*np.exp(-(x-(arg_4+-3.1480000000000246))**2/(2*(arg_5+0.0)**2))
	sum+=arg_2*np.exp(-(x-(arg_4+-1.1160000000000991))**2/(2*(arg_5+0.0)**2))
	sum+=arg_3*np.exp(-(x-arg_4)**2/(2*arg_5**2))
	sum+=arg_6*np.exp(-(x-(arg_8+-1.072999999999979))**2/(2*(arg_9+0.0)**2))
	sum+=arg_7*np.exp(-(x-arg_8)**2/(2*arg_9**2))
	sum+=arg_10*np.exp(-(x-(arg_8+1.8319999999999936))**2/(2*(arg_9+0.0)**2))
	mask_array_84595743 = (x > 760.546188872) & (x < 774.3936028720001);sub_x_84595743 = x[mask_array_84595743];sum[mask_array_84595743] += arg_11
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11):
	l=12
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_91857796 = np.exp(-(x-(arg_4+-4.024000000000001))**2/(2*(arg_5+0.0)**2))
	exp_26147253 = np.exp(-(x-(arg_4+-3.1480000000000246))**2/(2*(arg_5+0.0)**2))
	exp_30881199 = np.exp(-(x-(arg_4+-1.1160000000000991))**2/(2*(arg_5+0.0)**2))
	exp_8839065 = np.exp(-(x-arg_4)**2/(2*arg_5**2))
	exp_76994841 = np.exp(-(x-(arg_8+-1.072999999999979))**2/(2*(arg_9+0.0)**2))
	exp_12059642 = np.exp(-(x-arg_8)**2/(2*arg_9**2))
	exp_19002586 = np.exp(-(x-(arg_8+1.8319999999999936))**2/(2*(arg_9+0.0)**2))
	mask_array_74241733 = (x > 760.546188872) & (x < 774.3936028720001);sub_x_74241733 = x[mask_array_74241733]
	
	jac[:,0]= exp_91857796
	jac[:,1]= exp_26147253
	jac[:,2]= exp_30881199
	jac[:,3]= exp_8839065
	jac[:,4]= arg_0*(x-(arg_4+-4.024000000000001))   /((arg_5+0.0)**2) * exp_91857796 + arg_1*(x-(arg_4+-3.1480000000000246))   /((arg_5+0.0)**2) * exp_26147253 + arg_2*(x-(arg_4+-1.1160000000000991))   /((arg_5+0.0)**2) * exp_30881199 + arg_3*(x-arg_4)   /(arg_5**2) * exp_8839065
	jac[:,5]= arg_0*(x-(arg_4+-4.024000000000001))**2/((arg_5+0.0)**3) * exp_91857796 + arg_1*(x-(arg_4+-3.1480000000000246))**2/((arg_5+0.0)**3) * exp_26147253 + arg_2*(x-(arg_4+-1.1160000000000991))**2/((arg_5+0.0)**3) * exp_30881199 + arg_3*(x-arg_4)**2/(arg_5**3) * exp_8839065
	jac[:,6]= exp_76994841
	jac[:,7]= exp_12059642
	jac[:,8]= arg_6*(x-(arg_8+-1.072999999999979))   /((arg_9+0.0)**2) * exp_76994841 + arg_7*(x-arg_8)   /(arg_9**2) * exp_12059642 + arg_10*(x-(arg_8+1.8319999999999936))   /((arg_9+0.0)**2) * exp_19002586
	jac[:,9]= arg_6*(x-(arg_8+-1.072999999999979))**2/((arg_9+0.0)**3) * exp_76994841 + arg_7*(x-arg_8)**2/(arg_9**3) * exp_12059642 + arg_10*(x-(arg_8+1.8319999999999936))**2/((arg_9+0.0)**3) * exp_19002586
	jac[:,10]= exp_19002586
	jac[mask_array_74241733,11]= 1
	
	
	return jac
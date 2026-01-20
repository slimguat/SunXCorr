
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
	mask_array_19225639 = (x > 760.546188872) & (x < 774.3936028720001);sub_x_19225639 = x[mask_array_19225639];sum[mask_array_19225639] += arg_11
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11):
	l=12
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_59622473 = np.exp(-(x-(arg_4+-4.024000000000001))**2/(2*(arg_5+0.0)**2))
	exp_33022585 = np.exp(-(x-(arg_4+-3.1480000000000246))**2/(2*(arg_5+0.0)**2))
	exp_90239046 = np.exp(-(x-(arg_4+-1.1160000000000991))**2/(2*(arg_5+0.0)**2))
	exp_96094722 = np.exp(-(x-arg_4)**2/(2*arg_5**2))
	exp_20252183 = np.exp(-(x-(arg_8+-1.072999999999979))**2/(2*(arg_9+0.0)**2))
	exp_87750719 = np.exp(-(x-arg_8)**2/(2*arg_9**2))
	exp_18686522 = np.exp(-(x-(arg_8+1.8319999999999936))**2/(2*(arg_9+0.0)**2))
	mask_array_94959224 = (x > 760.546188872) & (x < 774.3936028720001);sub_x_94959224 = x[mask_array_94959224]
	
	jac[:,0]= exp_59622473
	jac[:,1]= exp_33022585
	jac[:,2]= exp_90239046
	jac[:,3]= exp_96094722
	jac[:,4]= arg_0*(x-(arg_4+-4.024000000000001))   /((arg_5+0.0)**2) * exp_59622473 + arg_1*(x-(arg_4+-3.1480000000000246))   /((arg_5+0.0)**2) * exp_33022585 + arg_2*(x-(arg_4+-1.1160000000000991))   /((arg_5+0.0)**2) * exp_90239046 + arg_3*(x-arg_4)   /(arg_5**2) * exp_96094722
	jac[:,5]= arg_0*(x-(arg_4+-4.024000000000001))**2/((arg_5+0.0)**3) * exp_59622473 + arg_1*(x-(arg_4+-3.1480000000000246))**2/((arg_5+0.0)**3) * exp_33022585 + arg_2*(x-(arg_4+-1.1160000000000991))**2/((arg_5+0.0)**3) * exp_90239046 + arg_3*(x-arg_4)**2/(arg_5**3) * exp_96094722
	jac[:,6]= exp_20252183
	jac[:,7]= exp_87750719
	jac[:,8]= arg_6*(x-(arg_8+-1.072999999999979))   /((arg_9+0.0)**2) * exp_20252183 + arg_7*(x-arg_8)   /(arg_9**2) * exp_87750719 + arg_10*(x-(arg_8+1.8319999999999936))   /((arg_9+0.0)**2) * exp_18686522
	jac[:,9]= arg_6*(x-(arg_8+-1.072999999999979))**2/((arg_9+0.0)**3) * exp_20252183 + arg_7*(x-arg_8)**2/(arg_9**3) * exp_87750719 + arg_10*(x-(arg_8+1.8319999999999936))**2/((arg_9+0.0)**3) * exp_18686522
	jac[:,10]= exp_18686522
	jac[mask_array_94959224,11]= 1
	
	
	return jac

# Function definition
import numpy as np
from numba import jit

def model_function(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	sum = np.zeros((len(x), ),dtype=np.float64)
	sum+=arg_0*np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	sum+=arg_1*np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	sum+=arg_2*np.exp(-(x-arg_3)**2/(2*arg_4**2))
	sum+=arg_5*np.exp(-(x-arg_6)**2/(2*arg_7**2))
	sum+=arg_8*np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	sum+=arg_9*np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	sum+=arg_10*np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_39694917 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_39694917 = x[mask_array_39694917];sum[mask_array_39694917] += arg_13
	mask_array_61117113 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_61117113 = x[mask_array_61117113];sum[mask_array_61117113] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_71846175 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_81505093 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_5811022 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_27402723 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_10541576 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_36225987 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_37193854 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_98075005 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_98075005 = x[mask_array_98075005]
	mask_array_26758065 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_26758065 = x[mask_array_26758065]
	
	jac[:,0]= exp_71846175
	jac[:,1]= exp_81505093
	jac[:,2]= exp_5811022
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_71846175 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_81505093 + arg_2*(x-arg_3)   /(arg_4**2) * exp_5811022
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_71846175 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_81505093 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_5811022
	jac[:,5]= exp_27402723
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_27402723 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_36225987
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_27402723 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_36225987
	jac[:,8]= exp_10541576
	jac[:,9]= exp_36225987
	jac[:,10]= exp_37193854
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_10541576 + arg_10*(x-arg_11)   /(arg_12**2) * exp_37193854
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_10541576 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_37193854
	jac[mask_array_98075005,13]= 1
	jac[mask_array_26758065,14]= 1
	
	
	return jac
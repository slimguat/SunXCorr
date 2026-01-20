
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
	mask_array_307876 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_307876 = x[mask_array_307876];sum[mask_array_307876] += arg_13
	mask_array_14684220 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_14684220 = x[mask_array_14684220];sum[mask_array_14684220] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_74949875 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_24369749 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_11220900 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_35457101 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_88568674 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_13742962 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_69187207 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_93415423 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_93415423 = x[mask_array_93415423]
	mask_array_58770300 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_58770300 = x[mask_array_58770300]
	
	jac[:,0]= exp_74949875
	jac[:,1]= exp_24369749
	jac[:,2]= exp_11220900
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_74949875 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_24369749 + arg_2*(x-arg_3)   /(arg_4**2) * exp_11220900
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_74949875 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_24369749 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_11220900
	jac[:,5]= exp_35457101
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_35457101 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_13742962
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_35457101 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_13742962
	jac[:,8]= exp_88568674
	jac[:,9]= exp_13742962
	jac[:,10]= exp_69187207
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_88568674 + arg_10*(x-arg_11)   /(arg_12**2) * exp_69187207
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_88568674 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_69187207
	jac[mask_array_93415423,13]= 1
	jac[mask_array_58770300,14]= 1
	
	
	return jac
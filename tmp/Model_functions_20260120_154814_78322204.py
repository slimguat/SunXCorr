
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
	mask_array_89441827 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_89441827 = x[mask_array_89441827];sum[mask_array_89441827] += arg_13
	mask_array_1570175 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_1570175 = x[mask_array_1570175];sum[mask_array_1570175] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_43120259 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_76900164 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_76055482 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_93187165 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_2634685 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_89079811 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_62598188 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_11100971 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_11100971 = x[mask_array_11100971]
	mask_array_89452902 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_89452902 = x[mask_array_89452902]
	
	jac[:,0]= exp_43120259
	jac[:,1]= exp_76900164
	jac[:,2]= exp_76055482
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_43120259 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_76900164 + arg_2*(x-arg_3)   /(arg_4**2) * exp_76055482
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_43120259 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_76900164 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_76055482
	jac[:,5]= exp_93187165
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_93187165 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_89079811
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_93187165 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_89079811
	jac[:,8]= exp_2634685
	jac[:,9]= exp_89079811
	jac[:,10]= exp_62598188
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_2634685 + arg_10*(x-arg_11)   /(arg_12**2) * exp_62598188
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_2634685 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_62598188
	jac[mask_array_11100971,13]= 1
	jac[mask_array_89452902,14]= 1
	
	
	return jac
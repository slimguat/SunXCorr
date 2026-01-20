
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
	mask_array_42358589 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_42358589 = x[mask_array_42358589];sum[mask_array_42358589] += arg_13
	mask_array_49134591 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_49134591 = x[mask_array_49134591];sum[mask_array_49134591] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_11133125 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_64256656 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_11935228 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_43371414 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_27905812 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_17341400 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_55317594 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_10448323 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_10448323 = x[mask_array_10448323]
	mask_array_53187697 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_53187697 = x[mask_array_53187697]
	
	jac[:,0]= exp_11133125
	jac[:,1]= exp_64256656
	jac[:,2]= exp_11935228
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_11133125 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_64256656 + arg_2*(x-arg_3)   /(arg_4**2) * exp_11935228
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_11133125 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_64256656 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_11935228
	jac[:,5]= exp_43371414
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_43371414 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_17341400
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_43371414 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_17341400
	jac[:,8]= exp_27905812
	jac[:,9]= exp_17341400
	jac[:,10]= exp_55317594
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_27905812 + arg_10*(x-arg_11)   /(arg_12**2) * exp_55317594
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_27905812 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_55317594
	jac[mask_array_10448323,13]= 1
	jac[mask_array_53187697,14]= 1
	
	
	return jac

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
	mask_array_28400938 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_28400938 = x[mask_array_28400938];sum[mask_array_28400938] += arg_13
	mask_array_18083078 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_18083078 = x[mask_array_18083078];sum[mask_array_18083078] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_9827286 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_1493550 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_30699588 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_31691905 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_4541 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_43023460 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_15633520 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_5789018 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_5789018 = x[mask_array_5789018]
	mask_array_85446210 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_85446210 = x[mask_array_85446210]
	
	jac[:,0]= exp_9827286
	jac[:,1]= exp_1493550
	jac[:,2]= exp_30699588
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_9827286 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_1493550 + arg_2*(x-arg_3)   /(arg_4**2) * exp_30699588
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_9827286 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_1493550 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_30699588
	jac[:,5]= exp_31691905
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_31691905 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_43023460
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_31691905 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_43023460
	jac[:,8]= exp_4541
	jac[:,9]= exp_43023460
	jac[:,10]= exp_15633520
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_4541 + arg_10*(x-arg_11)   /(arg_12**2) * exp_15633520
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_4541 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_15633520
	jac[mask_array_5789018,13]= 1
	jac[mask_array_85446210,14]= 1
	
	
	return jac
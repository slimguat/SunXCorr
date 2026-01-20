
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
	mask_array_34678082 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_34678082 = x[mask_array_34678082];sum[mask_array_34678082] += arg_13
	mask_array_34763266 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_34763266 = x[mask_array_34763266];sum[mask_array_34763266] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_45404197 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_98813277 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_55588171 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_87940724 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_25239614 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_60295870 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_89646295 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_15182914 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_15182914 = x[mask_array_15182914]
	mask_array_42423810 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_42423810 = x[mask_array_42423810]
	
	jac[:,0]= exp_45404197
	jac[:,1]= exp_98813277
	jac[:,2]= exp_55588171
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_45404197 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_98813277 + arg_2*(x-arg_3)   /(arg_4**2) * exp_55588171
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_45404197 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_98813277 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_55588171
	jac[:,5]= exp_87940724
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_87940724 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_60295870
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_87940724 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_60295870
	jac[:,8]= exp_25239614
	jac[:,9]= exp_60295870
	jac[:,10]= exp_89646295
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_25239614 + arg_10*(x-arg_11)   /(arg_12**2) * exp_89646295
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_25239614 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_89646295
	jac[mask_array_15182914,13]= 1
	jac[mask_array_42423810,14]= 1
	
	
	return jac
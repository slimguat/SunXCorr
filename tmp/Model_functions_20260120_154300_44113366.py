
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
	mask_array_64323252 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_64323252 = x[mask_array_64323252];sum[mask_array_64323252] += arg_13
	mask_array_81534491 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_81534491 = x[mask_array_81534491];sum[mask_array_81534491] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_88658324 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_54815553 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_37561301 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_23213706 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_53616714 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_12834109 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_84435044 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_32055599 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_32055599 = x[mask_array_32055599]
	mask_array_5829315 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_5829315 = x[mask_array_5829315]
	
	jac[:,0]= exp_88658324
	jac[:,1]= exp_54815553
	jac[:,2]= exp_37561301
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_88658324 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_54815553 + arg_2*(x-arg_3)   /(arg_4**2) * exp_37561301
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_88658324 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_54815553 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_37561301
	jac[:,5]= exp_23213706
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_23213706 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_12834109
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_23213706 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_12834109
	jac[:,8]= exp_53616714
	jac[:,9]= exp_12834109
	jac[:,10]= exp_84435044
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_53616714 + arg_10*(x-arg_11)   /(arg_12**2) * exp_84435044
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_53616714 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_84435044
	jac[mask_array_32055599,13]= 1
	jac[mask_array_5829315,14]= 1
	
	
	return jac
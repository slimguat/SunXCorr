
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
	mask_array_92507269 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_92507269 = x[mask_array_92507269];sum[mask_array_92507269] += arg_13
	mask_array_81521616 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_81521616 = x[mask_array_81521616];sum[mask_array_81521616] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_37078005 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_56359273 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_4999273 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_8641651 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_30509933 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_24056952 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_76409042 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_2881448 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_2881448 = x[mask_array_2881448]
	mask_array_55795521 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_55795521 = x[mask_array_55795521]
	
	jac[:,0]= exp_37078005
	jac[:,1]= exp_56359273
	jac[:,2]= exp_4999273
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_37078005 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_56359273 + arg_2*(x-arg_3)   /(arg_4**2) * exp_4999273
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_37078005 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_56359273 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_4999273
	jac[:,5]= exp_8641651
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_8641651 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_24056952
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_8641651 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_24056952
	jac[:,8]= exp_30509933
	jac[:,9]= exp_24056952
	jac[:,10]= exp_76409042
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_30509933 + arg_10*(x-arg_11)   /(arg_12**2) * exp_76409042
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_30509933 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_76409042
	jac[mask_array_2881448,13]= 1
	jac[mask_array_55795521,14]= 1
	
	
	return jac

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
	mask_array_66215767 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_66215767 = x[mask_array_66215767];sum[mask_array_66215767] += arg_13
	mask_array_80421981 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_80421981 = x[mask_array_80421981];sum[mask_array_80421981] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_8168956 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_11762578 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_18468602 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_35533903 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_52894891 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_14073727 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_19233964 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_98931595 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_98931595 = x[mask_array_98931595]
	mask_array_2580405 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_2580405 = x[mask_array_2580405]
	
	jac[:,0]= exp_8168956
	jac[:,1]= exp_11762578
	jac[:,2]= exp_18468602
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_8168956 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_11762578 + arg_2*(x-arg_3)   /(arg_4**2) * exp_18468602
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_8168956 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_11762578 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_18468602
	jac[:,5]= exp_35533903
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_35533903 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_14073727
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_35533903 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_14073727
	jac[:,8]= exp_52894891
	jac[:,9]= exp_14073727
	jac[:,10]= exp_19233964
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_52894891 + arg_10*(x-arg_11)   /(arg_12**2) * exp_19233964
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_52894891 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_19233964
	jac[mask_array_98931595,13]= 1
	jac[mask_array_2580405,14]= 1
	
	
	return jac
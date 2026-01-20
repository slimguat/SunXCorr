
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
	mask_array_32779860 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_32779860 = x[mask_array_32779860];sum[mask_array_32779860] += arg_13
	mask_array_59488830 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_59488830 = x[mask_array_59488830];sum[mask_array_59488830] += arg_14
	return sum





# Jacobian definition
import numpy as np
from numba import jit

def model_jacobian(x,arg_0,arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,arg_14):
	l=15
	if isinstance(x,float):x = np.array([x],dtype=np.float64)
	jac = np.zeros((len(x),l),dtype=np.float64)
	exp_42779251 = np.exp(-(x-(arg_3+-1.518000000000029))**2/(2*(arg_4+0.0)**2))
	exp_39854766 = np.exp(-(x-(arg_3+-0.9550000000000409))**2/(2*(arg_4+0.0)**2))
	exp_86881464 = np.exp(-(x-arg_3)**2/(2*arg_4**2))
	exp_50271820 = np.exp(-(x-arg_6)**2/(2*arg_7**2))
	exp_94099546 = np.exp(-(x-(arg_11+-1.82000000000005))**2/(2*(arg_12+0.0)**2))
	exp_87495016 = np.exp(-(x-(arg_6+43.48000000000002))**2/(2*(arg_7+0.0)**2))
	exp_78694406 = np.exp(-(x-arg_11)**2/(2*arg_12**2))
	mask_array_40740137 = (x > 700.670750872) & (x < 711.5926548720001);sub_x_40740137 = x[mask_array_40740137]
	mask_array_38207830 = (x > 745.626087872) & (x < 753.4274478719999);sub_x_38207830 = x[mask_array_38207830]
	
	jac[:,0]= exp_42779251
	jac[:,1]= exp_39854766
	jac[:,2]= exp_86881464
	jac[:,3]= arg_0*(x-(arg_3+-1.518000000000029))   /((arg_4+0.0)**2) * exp_42779251 + arg_1*(x-(arg_3+-0.9550000000000409))   /((arg_4+0.0)**2) * exp_39854766 + arg_2*(x-arg_3)   /(arg_4**2) * exp_86881464
	jac[:,4]= arg_0*(x-(arg_3+-1.518000000000029))**2/((arg_4+0.0)**3) * exp_42779251 + arg_1*(x-(arg_3+-0.9550000000000409))**2/((arg_4+0.0)**3) * exp_39854766 + arg_2*(x-arg_3)**2/(arg_4**3) * exp_86881464
	jac[:,5]= exp_50271820
	jac[:,6]= arg_5*(x-arg_6)   /(arg_7**2) * exp_50271820 + arg_9*(x-(arg_6+43.48000000000002))   /((arg_7+0.0)**2) * exp_87495016
	jac[:,7]= arg_5*(x-arg_6)**2/(arg_7**3) * exp_50271820 + arg_9*(x-(arg_6+43.48000000000002))**2/((arg_7+0.0)**3) * exp_87495016
	jac[:,8]= exp_94099546
	jac[:,9]= exp_87495016
	jac[:,10]= exp_78694406
	jac[:,11]= arg_8*(x-(arg_11+-1.82000000000005))   /((arg_12+0.0)**2) * exp_94099546 + arg_10*(x-arg_11)   /(arg_12**2) * exp_78694406
	jac[:,12]= arg_8*(x-(arg_11+-1.82000000000005))**2/((arg_12+0.0)**3) * exp_94099546 + arg_10*(x-arg_11)**2/(arg_12**3) * exp_78694406
	jac[mask_array_40740137,13]= 1
	jac[mask_array_38207830,14]= 1
	
	
	return jac
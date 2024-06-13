import numpy as np 
import pandas as pd
import albumentations as A
import warnings
warnings.filterwarnings("ignore")
#from pytorch_pretrained_bert.modeling import BertModel

def select_parameters_colour():
	hue_min = -15
	hue_max = 8

	sat_min = -20
	sat_max = 10

	val_min = -8
	val_max = 8


	p1 = np.random.uniform(hue_min,hue_max,1)
	p2 = np.random.uniform(sat_min,sat_max,1)
	p3 = np.random.uniform(val_min,val_max,1)

	return p1[0],p2[0],p3[0]

def select_rgb_shift():
	r_min = -10
	r_max = 10

	g_min = -10
	g_max = 10

	b_min = -10
	b_max = 10


	p1 = np.random.uniform(r_min,r_max,1)
	p2 = np.random.uniform(g_min,g_max,1)
	p3 = np.random.uniform(b_min,b_max,1)

	return p1[0],p2[0],p3[0]

def select_elastic_distorsion():
	sigma_min = 0
	sigma_max = 20

	alpha_affine_min = -20
	alpha_affine_max = 20

	p1 = np.random.uniform(sigma_min,sigma_max,1)
	p2 = np.random.uniform(alpha_affine_min,alpha_affine_max,1)

	return p1[0],p2[0]

def select_grid_distorsion():
	dist_min = 0
	dist_max = 0.2

	p1 = np.random.uniform(dist_min,dist_max,1)

	return p1[0]

def generate_transformer(prob = 0.5):
	list_operations = []
	probas = np.random.rand(7)

	if (probas[0]>prob):
		list_operations.append(A.VerticalFlip(always_apply=True))

	if (probas[1]>prob):
		list_operations.append(A.HorizontalFlip(always_apply=True))
	
	if (probas[2]>prob):
		p_rot = np.random.rand(1)[0]
		if (p_rot<=0.33):
			lim_rot = 90
		elif (p_rot>0.33 and p_rot<=0.66):
			lim_rot = 180
		else:
			lim_rot = 270
		list_operations.append(A.SafeRotate(always_apply=True, limit=(lim_rot,lim_rot+1e-4), interpolation=1, border_mode=4))
	
	if (probas[3]>prob):
		p1, p2, p3 = select_parameters_colour()
		list_operations.append(A.HueSaturationValue(always_apply=True,hue_shift_limit=(p1,p1+1e-4),sat_shift_limit=(p2,p2+1e-4),val_shift_limit=(p3,p3+1e-4)))
		
	pipeline_transform = A.Compose(list_operations)
	return pipeline_transform


if __name__ == "__main__":
	pass
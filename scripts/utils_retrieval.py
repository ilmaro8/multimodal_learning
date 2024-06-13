import numpy as np 
import warnings
warnings.filterwarnings("ignore")
from numba import jit

TEMPLATE = [[1,0,0,0,0],
			[0,1,0,0,0],
			[0,0,1,0,0],
			[0,0,0,1,0],
			[0,0,0,0,1],
			[1,1,0,0,0],
			[0,1,1,0,0],
			[0,0,1,1,0],
			[0,0,0,1,1],
			[1,1,1,0,0],
			[0,1,1,1,0],
			[0,0,1,1,1],
			[1,0,0,1,0],
			[1,0,1,0,0],
			[0,0,1,0,1],
			[0,1,0,1,0],
			[1,1,0,1,0],
			[0,1,0,0,1],
			[1,0,1,1,0]]

@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
	assert(u.shape[0] == v.shape[0])
	uv = 0
	uu = 0
	vv = 0
	for i in range(u.shape[0]):
		uv += u[i]*v[i]
		uu += u[i]*u[i]
		vv += v[i]*v[i]
	cos_theta = 1
	if uu!=0 and vv!=0:
		cos_theta = uv/np.sqrt(uu*vv)
	return cos_theta

def get_similar_sample(cls_to_analyze, dataset, n_top=5):
	
	similarities = []
	
	for r in dataset:
		
		sim = cosine_similarity_numba(cls_to_analyze, r)
		similarities.append(sim)
	
	#print(similarities)
	
	ind = np.argsort(similarities)[::-1][:n_top]
	
	return ind

def find_cls(filename_wsi, samples, data_cls):
	i = 0
	b = False
	cls_txt = None

	while(i<len(samples) and b == False):

		if (samples[i,0]==filename_wsi):
			#print(test_data_reports[i,0])
			b = True
			cls_txt = data_cls[i]
		else: 
			i = i + 1
	
	return cls_txt

def relevant(y_true, labels_found):
	
	relevant = 0
	
	for f in labels_found:
		
		if (np.array_equal(f, y_true)):
			
			relevant = relevant + 1
	
	return relevant


def relevant_micro(y_true, labels_found):
	
	relevant = 0
	tot_to_retrieve = 0

	idxs = []

	for i in range(len(y_true)):

		if (y_true[i]==1):
			idxs.append(i)

	elem_to_add = np.count_nonzero(y_true)

	for f in labels_found:
		
		#if (np.array_equal(f, y_true)):
		for i in idxs:

			if (f[i]==1):

				relevant = relevant + 1

		tot_to_retrieve = tot_to_retrieve + elem_to_add

	return relevant, tot_to_retrieve

def get_GTP(label, cont_classes):
	
	gtp = -1
	
	b = False
	i = 0
	
	while(b==False and i<len(TEMPLATE)):
		
		if (np.array_equal(TEMPLATE[i], label)==True):
			b = True
			gtp = cont_classes[i]
		else:
			i = i + 1
	
	return gtp


def average_precision(label, labels_found, cont_classes):
	
	gtp = get_GTP(label, cont_classes)
	
	relevant = 0
	
	tot_avg = 0.0
	
	for e, l in enumerate(labels_found):
		rel_n = 0
		
		if (np.array_equal(label, l)==True):
			
			relevant = relevant + 1
			rel_n = 1
		
		precision_n = relevant / (e + 1)
		elem = precision_n * rel_n
		tot_avg = tot_avg + (elem)
		
		#print(label, l, e, precision_n, rel_n, elem)
		
	return 1/gtp * tot_avg




def recall(y_true, relevant, cont_classes):
	
	i = 0
	b = False
	
	while(b==False and i<len(TEMPLATE)):
		
		if (np.array_equal(y_true, TEMPLATE[i])):
			
			b = True
			
		else:
			
			i = i + 1
	
	return relevant / cont_classes[i]

if __name__ == "__main__":
	pass
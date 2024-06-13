import numpy as np 
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import random
from torch.utils import data
from dataloader import Dataset_instance
import json
from glob import glob
import random
from enum_multi import PHASE

def generate_list_instances(instance_dir, filename):
	fname = os.path.split(filename)[-1]
	
	instance_csv = instance_dir+fname+'/'+fname+'_paths_densely_filter.csv'
	
	try:
		data_split = pd.read_csv(instance_csv, sep=',', header=None).values#[:10]
	except:
		instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'

	return instance_csv 


def get_features(instance_dir, ID, WEIGHTS):
	filename_features = instance_dir+ID+'/'+ID+'_features_'+WEIGHTS+'.npy'

	with open(filename_features, 'rb') as f:
		features = np.load(f)
		f.close()

	#"""
	n_indices = len(features)
	indices = np.random.choice(n_indices,n_indices,replace=False)
	
	#shuffled
	features = features[indices]
	#"""	
	return features

def get_dict_with_textual_reports(reports_folder):

	reports = dict()
	# get batch file paths
	batchfps = glob(reports_folder + '/*.json')
	# loop over batches and extract reports
	for batchfp in batchfps:
		if ('copy' not in batchfp and 'translated' not in batchfp and 'augmented_gpt' not in batchfp):
			with open(batchfp, 'r') as batchf:
				rbatch = json.load(batchf)

			tmp = {k: v for k, v in rbatch.items()}
			reports.update(tmp)

	dict_reports = dict()
		
	for rid, rdata in reports.items():
		#print(rid, rdata)
		try:
			slide_ids = rdata['slide_ids']
			
			for s in slide_ids:
				fname = rid + s
				#tmp = {fname: rbatch['diagnosis']}
				
				fname = fname.replace("/", "-")
				
				try:
					tmp = {fname: rdata['diagnosis']}
				except Exception as e:
					#print(e, fname, '2')
					tmp = {fname: rdata['diagnosis_nlp']}
					
				dict_reports.update(tmp)
			 
			fname = rid
			fname = fname.replace("/", "-")
				
			try:
				tmp = {fname: rdata['diagnosis']}
			except Exception as e:
				#print(e, fname, '2')
				tmp = {fname: rdata['diagnosis_nlp']}

			dict_reports.update(tmp)
				
		except Exception as e:
			#print(e, fname, '1')
			fname = rid
			
			if ('19/' in fname and '_' in fname):
				
				fname = fname.replace('_1','a')
				fname = fname.replace('_2','b')
				fname = fname.replace('_3','c')
			
			if ('.' in fname):
				fname = fname.split(".")[0]

			if ('_' in fname):
				fname = fname.split("_")[0]
			#tmp = {fname: rbatch['diagnosis']}
			
			fname = fname.replace("/", "-")

			#print(rdata)

			try:
				tmp = {fname: rdata['diagnosis']}
			except Exception as e:
				#print(e, fname, '2')
				tmp = {fname: rdata['diagnosis_nlp']}

				dict_reports.update(tmp)
				
	return dict_reports

def get_diagnosis(ID, dict_reports):
	#print(ID)
	fname_wsi = ID.split(".")[0]
	
	try:
		diagnosis = dict_reports[fname_wsi]
	except:
		try:
			fname_wsi = fname_wsi.lstrip('0')
			diagnosis = dict_reports[fname_wsi]
		except:
			diagnosis = 'none'


	return diagnosis

def get_augment_diagnosis(ID, translated, diagnosis_wsi):

	fname_wsi = ID.split(".")[0]
	flag_aug = False
	
	try:
		x = translated[fname_wsi]
		flag_aug = True
		
	except:
		try:
			fname_wsi = fname_wsi.lstrip('0')
			x = translated[fname_wsi]
			flag_aug = True
			
		except:
			diagnosis = diagnosis_wsi
	
	if (flag_aug):
		try:
			ks = list(x.keys())

			idx_su = random.randint(0, len(ks)-1)
			flag_aug = ks[idx_su]
			diagnosis = x[flag_aug]
		except:
			diagnosis = diagnosis_wsi
	return diagnosis


def get_generator_instances(csv_instances, mode, pipeline_transform, batch_size_instance):

	#csv_instances = pd.read_csv(input_bag_wsi, sep=',', header=None).values
		#number of instances
	n_elems = len(csv_instances)
	
		#params generator instances
	

	num_workers = int(n_elems/batch_size_instance) + 1

	if (n_elems > batch_size_instance):
		pin_memory = True
	else:
		pin_memory = False

	params_instance = {'batch_size': batch_size_instance,
			'num_workers': num_workers,
			'pin_memory': pin_memory,
			'shuffle': True}

	instances = Dataset_instance(csv_instances,mode,pipeline_transform)
	generator = data.DataLoader(instances, **params_instance)

	return generator

def get_splits_percentage(data, n, percentage, k=10):

	train_dataset = []
	valid_dataset = []

	if (n>=k):
		n = n%k

	train_p = []

	n_cont = n

	if (k==10):

		for p in range(percentage):

			train_p.append((n_cont + 1)%k)
			n_cont = n_cont + 1
			#train_p.append((n_cont + 1)%k)
			#n_cont = n_cont + 1

	else:
		
		for p in range(percentage):

			train_p.append((n_cont + 1)%k)
			n_cont = n_cont + 1

	print(train_p)

	for sample in data:

		fname = sample[0]
		cancer = sample[1]
		hgd = sample[2]
		lgd = sample[3]
		hyper = sample[4]
		normal = sample[5]
		f = sample[6]

		row = [fname, cancer, hgd, lgd, hyper, normal]

		if (f==n):

			valid_dataset.append(row)
		
		elif (f in train_p):
		
			train_dataset.append(row)

	train_dataset = np.array(train_dataset, dtype=object)#[:10]
	valid_dataset = np.array(valid_dataset, dtype=object)#[:10]

	return train_dataset, valid_dataset


def get_splits(data, n):
	
	train_dataset = []
	valid_dataset = []
	
	for sample in data:
		
		fname = sample[0]
		cancer = sample[1]
		hgd = sample[2]
		lgd = sample[3]
		hyper = sample[4]
		normal = sample[5]
		f = sample[6]
	
		row = [fname, cancer, hgd, lgd, hyper, normal]
		
		if (f==n):
			
			valid_dataset.append(row)
		
		else:
			
			train_dataset.append(row)
			
	train_dataset = np.array(train_dataset, dtype=object)
	valid_dataset = np.array(valid_dataset, dtype=object)
	
	return train_dataset, valid_dataset

def save_prediction(checkpoint_path, phase, epoch, arrays, DATASET):
	
	if (phase is PHASE.train):
		string_phase = 'train'
	elif (phase is PHASE.valid):
		string_phase = 'valid'
	elif (phase is PHASE.test):
		string_phase = 'test'
	
	if (phase is PHASE.test):
		storing_dir = checkpoint_path + '/' + string_phase + '/'
	else:
		storing_dir = checkpoint_path + '/' + string_phase + '/epoch_' + str(epoch) + '/metrics/'

	os.makedirs(storing_dir, exist_ok = True)

	if (phase is PHASE.test):
		filename_val = storing_dir+DATASET+'_predictions.csv'
	else:
		filename_val = storing_dir+'predictions.csv'
	
	File = {'filenames':arrays[:,0], 'pred_cancers':arrays[:,1], 'pred_hgd':arrays[:,2],'pred_lgd':arrays[:,3], 'pred_hyper':arrays[:,4], 'pred_normal':arrays[:,5]}
	df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper','pred_normal'])

	np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

def save_loss_function(checkpoint_path, phase, epoch, value):

	if (phase is PHASE.train):
		string_phase = 'train'
	elif (phase is PHASE.valid):
		string_phase = 'valid'
	elif (phase is PHASE.test):
		string_phase = 'test'
	
	storing_dir = checkpoint_path + '/' + string_phase + '/epoch_' + str(epoch) + '/'
	os.makedirs(storing_dir, exist_ok = True)

	filename_val = storing_dir+'loss_function.csv'
	array_val = [value]
	File = {'val':array_val}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

def save_hyperparameters(checkpoint_path, N_CLASSES, EMBEDDING_bool, lr):

	filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
	array_n_classes = [str(N_CLASSES)]
	array_lr = [str(lr)]
	array_embedding = [EMBEDDING_bool]
	File = {'n_classes':array_n_classes, 'lr':array_lr, 'embedding':array_embedding}


if __name__ == "__main__":
	pass
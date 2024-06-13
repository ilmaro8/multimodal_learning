import sys
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

import utils
import utils_retrieval
from dataloader import Dataset_bag_multilabel
from enum_multi import TYPE, AGGREGATION

argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

if torch.cuda.is_available():
	device = torch.device("cuda")
	print("working on gpu")
else:
	device = torch.device("cpu")
	print("working on cpu")
print(torch.backends.cudnn.version())

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-m', '--modality', help='multimodal/unimodal_txt/unimodal_img/contrastive',type=str, default='multimodal')
parser.add_argument('-a', '--algorithm', help='ABMIL, ADMIL, CLAM, DSMIL, transMIL',type=str, default='CLAM')
parser.add_argument('-b', '--data', help='images/reports',type=str, default='images')
parser.add_argument('-k', '--K', help='images/reports',type=int, default=-1)
parser.add_argument('-p', '--percentage', help='percentage % to use',type=int, default=-1)
parser.add_argument('-c', '--aggregation', help='normal/micro',type=str, default='micro')
parser.add_argument('-i', '--input_folder', help='path folder input csv (there will be a train.csv file including IDs and labels)',type=str, default='')
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-s', '--self_supervised', help='path folder with pretrained network',type=str, default='')
parser.add_argument('-d', '--DATA_FOLDER', help='path of the folder where to patches are stored',type=str, default='')
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')
parser.add_argument('-r', '--REPORT_FOLDER', help='folder where json files including textual reports are stored',type=str, default='True')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = 'resnet34'
MAGNIFICATION = '10'
PREPROCESSED_DATA = True
DATASET = 'test'
TEMPERATURE = 0.07
MIL_ALGORITHM = args.algorithm
MODALITY = args.modality

at_k = args.K
PERCENTAGE = args.percentage

type_data = args.data
TYPE_DATA = TYPE[type_data]

aggregation = args.aggregation
AGGREGATION_mode = AGGREGATION[aggregation]


hidden_space_len = 128

assert ((at_k == -1 and PERCENTAGE != -1) or (at_k !=- 1 and PERCENTAGE == -1))

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

INPUT_folder = args.input_folder
OUTPUT_folder = args.output_folder
SELF_folder = args.self_supervised
instance_dir = args.DATA_FOLDER
csv_folder = args.CSV_FOLDER

WEIGHTS = 'simclr'
model_weights_filename_pre_trained = args.self_supervised
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

#DIRECTORY BERT NEW MODEL
models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)

models_path = models_path+MODALITY+'/'
os.makedirs(models_path, exist_ok=True)

models_path = models_path+MIL_ALGORITHM+'/'
os.makedirs(models_path, exist_ok=True)

models_path = models_path+'magnification_'+str(MAGNIFICATION)+'x/'
os.makedirs(models_path, exist_ok = True)
models_path = models_path+'N_EXP_'+N_EXP_str+'/'
os.makedirs(models_path, exist_ok = True)

checkpoint_path = models_path+'checkpoints_MIL/'
os.makedirs(checkpoint_path, exist_ok = True)
retrieval_path = checkpoint_path+'retrieval/'
os.makedirs(retrieval_path, exist_ok = True)

if (AGGREGATION_mode is AGGREGATION.normal and PERCENTAGE == -1):
	filename_output = retrieval_path + 'mAP_at_'+str(at_k)+'_'+DATASET+'_'+type_data+'.csv'

elif (AGGREGATION_mode is AGGREGATION.normal and at_k == -1):
	filename_output = retrieval_path + 'mAP_percentage_'+str(PERCENTAGE)+'_'+DATASET+'_'+type_data+'.csv'

elif (AGGREGATION_mode is AGGREGATION.micro and PERCENTAGE == -1):
	filename_output = retrieval_path + 'mAP_micro_at_'+str(at_k)+'_'+DATASET+'_'+type_data+'.csv'

elif (AGGREGATION_mode is AGGREGATION.micro and at_k == -1):
	filename_output = retrieval_path + 'mAP_micro_percentage_'+str(PERCENTAGE)+'_'+DATASET+'_'+type_data+'.csv'

model_weights_filename = models_path+'model.pt'
model_weights_filename_checkpoint = models_path+'checkpoint.pt'


#CSV LOADING
print("CSV LOADING ")
k = 10
N_CLASSES = 5

LABELS = 'GT'

csv_filename_k_folds = csv_folder + str(k)+ '_cross_validation_main.csv'

#read data
data_split = pd.read_csv(csv_filename_k_folds, sep=',', header=None).values#[:10]

train_dataset, valid_dataset = utils.get_splits(data_split, N_EXP)


csv_filename_testing =  csv_folder+'ground_truth_test.csv'
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

list_training_samples = train_dataset[:,0]

#ALL DATA ARE DATA TO USE
samples = np.append(train_dataset, valid_dataset, axis=0)
samples = np.append(samples, test_dataset, axis=0)

n_samples = len(samples)
#read data

#ELEM TO ANALYZE
if (at_k != -1):
	num_elem_to_retrieve = at_k

if (PERCENTAGE != -1):
	num_elem_to_retrieve = int(PERCENTAGE / 100 * n_samples)

reports_folder = args.REPORT_FOLDER

len_max_seq = 512

dict_reports = utils.get_dict_with_textual_reports(reports_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size_bag = 1

params_valid_bag = {'batch_size': 1,
		  'shuffle': False}


testing_set_bag = Dataset_bag_multilabel(samples[:,0], samples[:,1:])
testing_generator_bag = data.DataLoader(testing_set_bag, **params_valid_bag)

print("initialize CNN")

try:
	bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
	tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
except:
	try:
		bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
		tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
	except:
		bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
		tokenizer = AutoTokenizer.from_pretrained(bert_chosen) 

###LOAD embeddings
test_reports_embeddings_fname = checkpoint_path + '/embeddings/cls_reports.npy'
test_images_embeddings_fname = checkpoint_path + '/embeddings/cls_images.npy'
test_labels_fname = checkpoint_path + '/embeddings/labels.csv'

n_elems_tot = 0

test_data = pd.read_csv(test_labels_fname, sep=',', header=None).values
filenames = test_data[:,0]
labels_cumulative = test_data[:,1:]

with open(test_images_embeddings_fname, 'rb') as f:
		cls_wsis = np.load(f)

with open(test_reports_embeddings_fname, 'rb') as f:
	cls_reports = np.load(f)


###### occurences of samples per label

CONT_CLASSES = [0 for i in range(len(utils_retrieval.TEMPLATE))]

for x, wsi in enumerate(samples):
	label = wsi[1:]
	
	i = 0
	b = False
	
	while(b==False and i<len(utils_retrieval.TEMPLATE)):
		
		if (np.array_equal(label, utils_retrieval.TEMPLATE[i])):
			
			b = True
			CONT_CLASSES[i] = CONT_CLASSES[i] + 1
			
		else:
			
			i = i + 1
		
	if (b==False):
		print(label)


def get_precision(DATASET, test_dataset):

	maps = []
	precisions = []
	recalls = []

	for wsi in test_dataset:
		
		fname = wsi[0]
		label = wsi[1:]
		diagnosis = utils.get_diagnosis(fname, dict_reports)

		if (TYPE_DATA is TYPE.reports):

			cls_report = utils_retrieval.find_cls(fname, samples, cls_reports)
			idxs = utils_retrieval.get_similar_sample(cls_report, cls_wsis, num_elem_to_retrieve)

		elif (TYPE_DATA is TYPE.images):
			cls_wsi = utils_retrieval.find_cls(fname, samples, cls_wsis)
			idxs = utils_retrieval.get_similar_sample(cls_wsi, cls_reports, num_elem_to_retrieve)
		
		labels_found = []

		for i in idxs:

			fname_sim = samples[i,0]
			label_sim = samples[i,1:]
			labels_found = np.append(labels_found, label_sim, axis=0)
		
		labels_found = np.reshape(labels_found, (int(len(labels_found)/5), 5))
		dist_tot = 0

		if (AGGREGATION_mode is AGGREGATION.normal):
			cont = 0
			d = utils_retrieval.average_precision(label, labels_found, CONT_CLASSES)
			#print(fname, d)
			maps.append(d)
			dist_tot = dist_tot + d
			cont = cont + 1

		elif (AGGREGATION_mode is AGGREGATION.micro):

			cont = 0
			d = utils_retrieval.average_precision(label, labels_found, CONT_CLASSES)
			#print(fname, d)
			maps.append(d)
			dist_tot = dist_tot + d
			cont = cont + 1	

	tot_map = np.sum(maps) / len(maps)
	print("tot_map " + str(DATASET) + ": " + str(tot_map))


	array = [tot_map]
	File = {'val':array}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename_output, df.values, fmt='%s',delimiter=',')

get_precision(DATASET, test_dataset)

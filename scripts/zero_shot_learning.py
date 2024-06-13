import sys
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

import utils
from dataloader import Dataset_bag_multilabel
from enum_multi import MOD, PHASE
from utils_retrieval import cosine_similarity_numba

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
parser.add_argument('-t', '--temperature', help='batch size bags',type=float, default=0.07)
parser.add_argument('-a', '--algorithm', help='ABMIL, ADMIL, CLAM, DSMIL, transMIL',type=str, default='CLAM')
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
TEMPERATURE = args.temperature
MIL_ALGORITHM = args.algorithm
MODALITY = args.modality

if (MODALITY is MOD.contrastive or MODALITY is MOD.contrastive_clip):
	from model_contrastive import MultimodalArchitecture
else:
	from model import MultimodalArchitecture

hidden_dim = 128
input_dim = 512

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

#TODO modify before uploading
INPUT_folder = args.input_folder
OUTPUT_folder = args.output_folder
SELF_folder = args.self_supervised
instance_dir = args.DATA_FOLDER
csv_folder = args.CSV_FOLDER

print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

WEIGHTS = 'simclr'
model_weights_filename_pre_trained = args.self_supervised

models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)

models_path = models_path+MODALITY+'/'
os.makedirs(models_path, exist_ok=True)

if ('branch' not in MODALITY):

	models_path = models_path+MIL_ALGORITHM+'/'
	os.makedirs(models_path, exist_ok=True)

models_path = models_path+'magnification_'+str(MAGNIFICATION)+'x/'
os.makedirs(models_path, exist_ok = True)
models_path = models_path+'N_EXP_'+N_EXP_str+'/'
os.makedirs(models_path, exist_ok = True)
checkpoint_path = models_path+'checkpoints_MIL/zero_shot_learning/'
os.makedirs(checkpoint_path, exist_ok = True)

model_weights_filename = models_path+'model.pt'

N_CLASSES = 5

zero_shot_folder = checkpoint_path+'wsis/'
os.makedirs(zero_shot_folder, exist_ok = True)


cls_concepts_fname = checkpoint_path + 'cls_reports.npy'

with open(cls_concepts_fname, 'rb') as f:
	cls_concepts = np.load(f)
	f.close()

cls_concepts_fname = checkpoint_path + 'concepts.csv'
concepts_name = pd.read_csv(cls_concepts_fname, sep=',', header=None).values


#CSV LOADING
print("CSV LOADING ")
k = 10
N_CLASSES = 5

LABELS = 'GT'

csv_filename_testing = csv_folder+'ground_truth_testing.csv'
samples = pd.read_csv(csv_filename_testing, sep=',', header=None).values

n_samples = len(samples)


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


model = MultimodalArchitecture(device, CNN_TO_USE, MODALITY, MIL_ALGORITHM, N_CLASSES, input_dim, hidden_dim, TEMPERATURE)

model.load_state_dict(torch.load(model_weights_filename), strict=False)
model.to(device)
model.eval()

iterations = len(samples)
dataloader_iterator = iter(testing_generator_bag)

phase = 'test'
phase = PHASE[phase]

cls_wsis = []
cls_reports = []

filenames = []
labels_cumulative = []

with torch.no_grad():
	j = 0

	for i in range(iterations):

		filenames = []
		similarities = []
		classes = []

		print('%d / %d ' % (i, iterations))
		try:
			ID, labels = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(generator)
			ID, labels = next(dataloader_iterator)
			#inputs: bags, labels: labels of the bags

		ID = ID[0]
		filename_wsi = ID
		#"""

		print(filename_wsi)

		csv_filename = utils.generate_list_instances(instance_dir, filename_wsi)
		csv_filename = pd.read_csv(csv_filename, sep=',', header=None).values

		filename_features = instance_dir+filename_wsi+'/'+filename_wsi+'_features_'+WEIGHTS+'.npy'

		with open(filename_features, 'rb') as f:
			features_np = np.load(f)
			f.close()

		for p, patch in enumerate(features_np):
			
			inputs_embedding = torch.tensor(patch, requires_grad=False).float().to(device, non_blocking=True).unsqueeze(0)

			_, embeddings_img, _, _, _ = model(inputs_embedding, None, None, phase)
			patch_features = embeddings_img.cpu().data.numpy()
			#patch_features = np.expand_dims(patch_features, 0)
			#print(patch_features.shape)

			max_sim = -1
			current_idx = -1
			current_fname = csv_filename[p,0]
			current_cls = patch_features

			for c in range(len(concepts_name)):
				
				cls_concept = cls_concepts[c]
				sim = cosine_similarity_numba(current_cls, cls_concept)
				
				if (sim >= max_sim):
					max_sim = sim
					current_idx = c
			
			filenames.append(current_fname)
			similarities.append(max_sim)
			classes.append(concepts_name[current_idx,0])	

		#save sample
		sample_filename = zero_shot_folder+filename_wsi+'.csv'
		
		#File = {'filenames':filenames, 'pred_cancers':labels_cumulative[:,0], 'pred_hgd':labels_cumulative[:,1],'pred_lgd':labels_cumulative[:,2], 'pred_hyper':labels_cumulative[:,3], 'pred_normal':labels_cumulative[:,4]}
		File = {'filenames':filenames, 'similarities':similarities, 'classes':classes}

		df = pd.DataFrame(File,columns=['filenames','similarities','classes'])
		np.savetxt(sample_filename, df.values, fmt='%s',delimiter=',')	
		#save filename

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

####PATH where find patches
WEIGHTS = 'simclr'
model_weights_filename_pre_trained = args.self_supervised

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
checkpoint_path = models_path+'checkpoints_MIL/embeddings/'
os.makedirs(checkpoint_path, exist_ok = True)

model_weights_filename = models_path+'model.pt'

test_reports_embeddings_fname = checkpoint_path + 'cls_reports.npy'
test_images_embeddings_fname = checkpoint_path + 'cls_images.npy'
test_labels_fname = checkpoint_path + 'labels.csv'

N_CLASSES = 5
	

#CSV LOADING
print("CSV LOADING ")
k = 10
N_CLASSES = 5

LABELS = 'GT'

csv_filename_k_folds = csv_folder + str(k)+ '_cross_validation_main.csv'

#read data
data_split = pd.read_csv(csv_filename_k_folds, sep=',', header=None).values#[:10]

train_dataset, valid_dataset = utils.get_splits(data_split, N_EXP)

csv_filename_testing = csv_folder+'ground_truth_testing.csv'
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

list_training_samples = train_dataset[:,0]

#ALL DATA ARE DATA TO USE
samples = np.append(train_dataset, valid_dataset, axis=0)
samples = np.append(samples, test_dataset, axis=0)

n_samples = len(samples)

reports_folder = args.REPORT_FOLDER

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
		diagnosis_wsi = utils.get_diagnosis(filename_wsi, dict_reports)

		encoded_text = tokenizer.encode_plus(
						diagnosis_wsi,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						pad_to_max_length=False
						)
		#"""
		label_wsi = labels[0].cpu().data.numpy().flatten()

		print("[" + str(i) + "/" + str(iterations) + "], " + "inputs_bag: " + str(filename_wsi))
		print("labels: " + str(label_wsi))

		features_np = utils.get_features(instance_dir, filename_wsi, WEIGHTS)

		inputs_embedding = torch.as_tensor(features_np).float().to(device, non_blocking=True)

		_, cls_img, _, cls_txt, _ = model(inputs_embedding, encoded_text, None, phase)

			
		#sigmoid_output_mm = F.sigmoid(logits_mm)
		cls_txt_np = cls_txt.cpu().data.numpy()
		cls_img_np = cls_img.cpu().data.numpy()

		filenames = np.append(filenames, filename_wsi)
		labels_cumulative = np.append(labels_cumulative,label_wsi)

		cls_reports = np.append(cls_reports, cls_txt_np)
		cls_wsis = np.append(cls_wsis, cls_img_np)

cls_reports = np.reshape(cls_reports,(n_samples,hidden_dim))
cls_wsis = np.reshape(cls_wsis,(n_samples,hidden_dim))

labels_cumulative = np.reshape(labels_cumulative, (n_samples, 5))

with open(test_reports_embeddings_fname, 'wb') as f:
	np.save(f, cls_reports)

with open(test_images_embeddings_fname, 'wb') as f:
	np.save(f, cls_wsis)

#File = {'filenames':filenames, 'pred_cancers':labels_cumulative[:,0], 'pred_hgd':labels_cumulative[:,1],'pred_lgd':labels_cumulative[:,2], 'pred_hyper':labels_cumulative[:,3], 'pred_normal':labels_cumulative[:,4]}
File = {'filenames':samples[:,0], 'pred_cancers':samples[:,1], 'pred_hgd':samples[:,2],'pred_lgd':samples[:,3], 'pred_hyper':samples[:,4], 'pred_normal':samples[:,5]}

df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper','pred_normal'])
np.savetxt(test_labels_fname, df.values, fmt='%s',delimiter=',')	
#save filename

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
from model import MultimodalArchitecture

import utils
from dataloader import Dataset_bag_multilabel
from data_augmentation import generate_transformer
from metrics_multilabel import accuracy_micro, f1_scores, precisions, recalls
import json
from enum_multi import PHASE

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
parser.add_argument('-p', '--preprocessed', help='pre-processed data: True False',type=str, default='True')
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
EMBEDDING_bool = 'True'
PREPROCESSED_DATA = args.preprocessed
TEMPERATURE = args.temperature
MIL_ALGORITHM = args.algorithm
MODALITY = args.modality
DATASET = 'test'

if (EMBEDDING_bool=='True'):
	EMBEDDING_bool = True
else:
	EMBEDDING_bool = False

if (PREPROCESSED_DATA=='True'):
	PREPROCESSED_DATA = True
else:
	PREPROCESSED_DATA = False

hidden_space_len = 128

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

####PATH where find patches

print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

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
checkpoint_path = models_path+'checkpoints_MIL/'
os.makedirs(checkpoint_path, exist_ok = True)

model_weights_filename = models_path+'model.pt'

N_CLASSES = 5
	
print("CSV LOADING ")

csv_filename_testing = csv_folder+'ground_truth_testing.csv'
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

print("Load reports: start")
reports_folder = args.REPORT_FOLDER

try:
	translated_reports = reports_folder + "translated_augmented_gpt_filtered.json"
	with open(translated_reports, 'r') as batchf:
		translated = json.load(batchf)
except:
	translated_reports = reports_folder + "translated_augmented.json"
	with open(translated_reports, 'r') as batchf:
		translated = json.load(batchf)

print("Load reports: over")

dict_reports = utils.get_dict_with_textual_reports(reports_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size_bag = 1

params_valid_bag = {'batch_size': 1,
		  'shuffle': False}


testing_set_bag = Dataset_bag_multilabel(test_dataset[:,0], test_dataset[:,1:])
testing_generator_bag = data.DataLoader(testing_set_bag, **params_valid_bag)

print("initialize CNN")
input_dim = 512
hidden_dim = 128

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

print("initialize hyperparameters")
import torch.optim as optim



print("testing")
print("testing at WSI level")
y_pred = []
y_true = []

phase = 'test'
phase = PHASE[phase]

filenames_wsis = []
pred_cancers = []
pred_hgd = []
pred_lgd = []
pred_hyper = []
pred_normal = []

model.eval()

iterations = len(test_dataset)
dataloader_iterator = iter(testing_generator_bag)

with torch.no_grad():
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
		"""
		diagnosis_wsi = utils.get_diagnosis(filename_wsi, dict_reports)

		encoded_text = tokenizer.encode_plus(
						diagnosis_wsi,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						pad_to_max_length=False
						)
		"""
		label_wsi = labels[0].cpu().data.numpy().flatten()
		labels_local = labels.float().flatten().to(device, non_blocking=True)

		print("[" + str(i) + "/" + str(iterations) + "], " + "inputs_bag: " + str(filename_wsi))
		print("labels: " + str(label_wsi))

		features_np = utils.get_features(instance_dir, filename_wsi, WEIGHTS)

		inputs_embedding = torch.as_tensor(features_np).float().to(device, non_blocking=True)

		logits_img, cls_img, logits_txt, cls_txt, _ = model(inputs_embedding, None, None, phase)

		sigmoid_output_img = F.sigmoid(logits_img)
		outputs_wsi_np_img = sigmoid_output_img.cpu().data.numpy()
		
		print()
		print("pred_img: " + str(outputs_wsi_np_img))
		print()

		output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

		filenames_wsis = np.append(filenames_wsis, filename_wsi)
		pred_cancers = np.append(pred_cancers, outputs_wsi_np_img[0])
		pred_hgd = np.append(pred_hgd, outputs_wsi_np_img[1])
		pred_lgd = np.append(pred_lgd, outputs_wsi_np_img[2])
		pred_hyper = np.append(pred_hyper, outputs_wsi_np_img[3])
		pred_normal = np.append(pred_normal, outputs_wsi_np_img[4])


		y_pred = np.append(y_pred,output_norm)
		y_true = np.append(y_true,label_wsi)


preds = np.stack((filenames_wsis, pred_cancers, pred_hgd, pred_lgd, pred_hyper, pred_normal), axis=1)


utils.save_prediction(checkpoint_path, phase, None, preds, DATASET)

micro_accuracy = accuracy_micro(y_true, y_pred, None, None, None, DATASET)
f1_score_macro, f1_score_micro, f1_score_weighted = f1_scores(y_true, y_pred, i, N_CLASSES, checkpoint_path, phase, None, DATASET)
precision_score_macro, precision_score_micro = precisions(y_true, y_pred, i, N_CLASSES, checkpoint_path, phase, None, DATASET)
recall_score_macro, recall_score_micro = recalls(y_true, y_pred, i, N_CLASSES, checkpoint_path, phase, None, DATASET)

print("micro_accuracy " + str(micro_accuracy)) 
print("f1_score_macro " + str(f1_score_macro))
print("f1_score_micro " + str(f1_score_micro))
print("f1_score_weighted " + str(f1_score_weighted))
print("precision_score_macro " + str(precision_score_macro))
print("precision_score_micro " + str(precision_score_micro))
print("recall_score_macro " + str(recall_score_macro))
print("recall_score_micro " + str(recall_score_micro))


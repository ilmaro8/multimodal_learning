import sys
import torch
import numpy as np
import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

import utils
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

####PATH where find patches
instance_dir = '/home/niccolo/ExamodePipeline/Colon_WSI_patches/magnifications/'+'magnification_'+MAGNIFICATION+'x/'

print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

WEIGHTS = 'simCLR'
WEIGHTS = '5'

models_path = '/home/niccolo/ExamodePipeline/multimodal/Colon/models_weights/'
os.makedirs(models_path, exist_ok=True)

models_path = models_path+MODALITY+'/'
os.makedirs(models_path, exist_ok=True)

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

CONCEPTS = ['Adenocarcinoma',
			'Adenocarcinoma (Invasive adenocarcinoma)',
			'Adenoma',
			'Dysplasia',
			'High Grade Dysplasia',
			'Hyperplastic Polyp',
			'Mild Dysplasia',
			'Moderate Dysplasia',
			'Polyp',
			'Pre-Cancerous Dysplasia',
			'Serrated Adenoma',
			'Severe Dysplasia',
			'Tubular Adenoma',
			'Tubulovillous Adenoma',
			'Villous Adenoma']

reports_folder = '/home/niccolo/ExamodePipeline/multimodal/Colon/csv_folder/reports/'

dict_reports = utils.get_dict_with_textual_reports(reports_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size_bag = 1

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

phase = 'test'
phase = PHASE[phase]

cls_wsis = []
cls_reports = []

filenames = []
labels_cumulative = []

with torch.no_grad():
	j = 0

	for concept in CONCEPTS:
		print(concept)
		
		concept = concept.lower()

		encoded_text = tokenizer.encode_plus(
						concept,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						pad_to_max_length=False
						)
		
		_, _, _, cls_txt, _ = model(None, encoded_text, None, phase)

			
		#sigmoid_output_mm = F.sigmoid(logits_mm)
		cls_txt_np = cls_txt.cpu().data.numpy()

		cls_reports = np.append(cls_reports, cls_txt_np)

cls_reports = np.reshape(cls_reports,(len(CONCEPTS),hidden_dim))


test_reports_embeddings_fname = checkpoint_path + 'cls_reports.npy'

with open(test_reports_embeddings_fname, 'wb') as f:
	np.save(f, cls_reports)


list_concepts = checkpoint_path + 'concepts.csv'

File = {'filenames':CONCEPTS}

df = pd.DataFrame(File,columns=['filenames'])
np.savetxt(list_concepts, df.values, fmt='%s',delimiter=',')	
#save filename

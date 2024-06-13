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
from model_contrastive import MultimodalArchitecture
from transformers import AutoTokenizer


import json
import losses

import utils
from dataloader import Dataset_bag_multilabel, Balanced_Multimodal
from data_augmentation import generate_transformer
from enum_multi import MOD, ALG, PHASE

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
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=15)
parser.add_argument('-p', '--preprocessed', help='pre-processed data: True False',type=str, default='True')
parser.add_argument('-m', '--modality', help='multimodal/unimodal_txt/unimodal/contrastive',type=str, default='multimodal')
parser.add_argument('-b', '--batch', help='batch size bags',type=int, default=4)
parser.add_argument('-t', '--temperature', help='batch size bags',type=float, default=0.07)
parser.add_argument('-a', '--algorithm', help='ABMIL, ADMIL, CLAM, DSMIL, transMIL',type=str, default='ADMIL')
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
BATCH_SIZE = 512
BATCH_SIZE_str = str(BATCH_SIZE)
pool_algorithm = 'att'
TASK = 'multilabel'
MAGNIFICATION = '10'
EMBEDDING_bool = 'True'
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
GATED_bool = False
PREPROCESSED_DATA = args.preprocessed
FLAG_CLASSIFIER = False
#BERT_PRE_TRAIN = args.bert
BERT_PRE_TRAIN = 'pubmed'
MODALITY = args.modality
BATCH_SIZE_bags = args.batch
TEMPERATURE = args.temperature
MIL_ALGORITHM = args.algorithm

EMBEDDING_bool = True

if (PREPROCESSED_DATA=='True'):
	PREPROCESSED_DATA = True
else:
	PREPROCESSED_DATA = False

hidden_space_len = 128

#TODO modify before uploading
INPUT_folder = args.input_folder
OUTPUT_folder = args.output_folder
SELF_folder = args.self_supervised
instance_dir = args.DATA_FOLDER
csv_folder = args.CSV_FOLDER


seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


####PATH where find patches

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


if ('contrastive' in MODALITY):
	elem_to_discard = len(train_dataset) % BATCH_SIZE_bags
	train_dataset = train_dataset[elem_to_discard:]
	elem_to_discard = len(valid_dataset) % BATCH_SIZE_bags
	valid_dataset = valid_dataset[elem_to_discard:]

csv_filename_testing = csv_folder+'ground_truth_testing.csv'
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

list_training_samples = train_dataset[:,0]

#list_training_samples = np.append(data_split[:,0], test_dataset[:,0],axis=0)
all_data = np.append(train_dataset, valid_dataset, axis=0)
all_data = np.append(all_data, test_dataset, axis=0)

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

AUGMENT_PROB_REPORT = 0.5
AUGMENT_PROB_THRESHOLD = 0.5
prob = AUGMENT_PROB_THRESHOLD

sampler = Balanced_Multimodal

if ('contrastive' in MODALITY):
	BATCH_SIZE_bags_valid = 4
else:
	BATCH_SIZE_bags_valid = 1

if (PREPROCESSED_DATA==True):
	params_train_bag = {'batch_size': BATCH_SIZE_bags,
		#'sampler': sampler(train_dataset,alpha=0.25)}
		'drop_last':True,
		'shuffle': True}
else:
	params_train_bag = {'batch_size': BATCH_SIZE_bags,
		'sampler': sampler(train_dataset,alpha=0.25),
		'drop_last':True}

params_valid_bag = {'batch_size': BATCH_SIZE_bags_valid,
		'drop_last':True,
		'shuffle': True}


training_set_bag = Dataset_bag_multilabel(train_dataset[:,0], train_dataset[:,1:])
training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

validation_set_bag = Dataset_bag_multilabel(valid_dataset[:,0], valid_dataset[:,1:])
validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)

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
model.conv_layers.load_state_dict(torch.load(model_weights_filename_pre_trained), strict=False)
#model.load_state_dict(torch.load(model_weights_filename_MoCo), strict=False)
model.eval()
model.to(device)

num_epochs = EPOCHS

# assign enum
MIL_ALGORITHM = ALG[MIL_ALGORITHM]
MODALITY = MOD[MODALITY]


print("initialize hyperparameters")
import torch.optim as optim

if (MIL_ALGORITHM is ALG.CLAM_MB or MIL_ALGORITHM is ALG.CLAM_SB):
	lr = 2e-4
	wt_decay = 1e-5
else:
	lr = 1e-4
	wt_decay = 1e-4

optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=True)

####CRITERIONs
criterion = torch.nn.BCEWithLogitsLoss()
criterion_representation_cosine = torch.nn.CosineEmbeddingLoss()
#criterion_representation_rmse = RMSELoss()
#criterion_representation_rmse = torch.nn.MSELoss()
criterion_representation_rmse = torch.nn.L1Loss()

if (MODALITY is MOD.contrastive_clip or MODALITY is MOD.multimodal_clip):
	criterion_contrastive = losses.CLIP_Loss(batch_size = BATCH_SIZE_bags, temperature = TEMPERATURE)
else:	
	criterion_contrastive = losses.NT_Xent(batch_size = BATCH_SIZE_bags, temperature = TEMPERATURE)

criterion_representation_pairwise = losses.ContrastiveLoss()

def forward_features(generator_instance):

	features = []
	with torch.no_grad():
		for instances in generator_instance:
			instances = instances.to(device, non_blocking=True)

			# forward + backward + optimize
			feats = model.conv_layers(instances)
			feats = feats.view(-1, input_dim)
			feats_np = feats.cpu().data.numpy()
			
			features = np.append(features,feats_np)
	
	return features

def compute_features(instance_dir, ID):
	pipeline_transform = generate_transformer()
	instances_filename_sample = utils.generate_list_instances(instance_dir, ID)

	csv_instances = pd.read_csv(instances_filename_sample, sep=',', header=None).values
	n_elems = len(csv_instances)

	generator_instance = utils.get_generator_instances(csv_instances, 'train', pipeline_transform, batch_size_instance)
	model.eval()

	features = forward_features(generator_instance)
			
	features_np = np.reshape(features,(n_elems,input_dim))		

	return features_np

def get_predictions_embeddings(input_img, input_txt, label, phase):
	
	logits_img = None
	cls_img = None
	logits_txt = None
	cls_txt = None
	encoded_text = None

	if (phase is PHASE.train):
		flag_gradients = True
	else:
		flag_gradients = False

	inputs_embedding = None
	encoded_text = None

	if (input_txt is not None):
		
		encoded_text = tokenizer.encode_plus(
			input_txt,
			add_special_tokens=True,
			return_token_type_ids=True,
			return_attention_mask=True,
			pad_to_max_length=False
			)
		txt_len = len(encoded_text['input_ids'])
		

	if (input_img is not None):
		
		inputs_embedding = torch.tensor(input_img, requires_grad=flag_gradients).float().to(device, non_blocking=True)

	logits_img, embeddings_img, logits_txt, embeddings_txt, total_inst_loss = model(inputs_embedding, encoded_text, label, phase)

	return logits_img, embeddings_img, logits_txt, embeddings_txt, total_inst_loss

epoch = 0

best_loss = 100000.0
#number of epochs without improvement
EARLY_STOP_NUM = 10
early_stop_cont = 0
epoch = 0
LAMBDA_INSTANCE = 0.2
EPOCH_THRESH_CLAM = 0

batch_size_instance = int(BATCH_SIZE_str)
label_embedding_pos = torch.tensor([1]).to(device)

def evaluate(epoch):
	
	phase = 'valid'
	phase = PHASE[phase]

	total_loss = 0.0

	img_loss = 0.0
	txt_loss = 0.0
	instance_loss = 0.0
	representation_loss_rmse = 0.0
	representation_loss_cosine = 0.0
	contrastive_loss = 0.0
	representation_pairwise = 0.0

	filenames_wsis_i = []
	pred_cancers_i = []
	pred_hgd_i = []
	pred_lgd_i = []
	pred_hyper_i = []
	pred_normal_i = []

	y_pred_i = []
	y_true_i = []

	filenames_wsis_t = []
	pred_cancers_t = []
	pred_hgd_t = []
	pred_lgd_t = []
	pred_hyper_t = []
	pred_normal_t = []

	y_pred_t = []
	y_true_t = []

	generator = validation_generator_bag
	dataloader_iterator = iter(validation_generator_bag)
	iterations = int(len(valid_dataset) / BATCH_SIZE_bags_valid)#+1
	batch_size_bag = BATCH_SIZE_bags_valid
	model.eval()
	
	for i in range(iterations):
		
		print('[%d], %d / %d ' % (epoch, i, iterations))
		try:
			IDs, labels = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(generator)
			IDs, labels = next(dataloader_iterator)
		
		array_logits_img = []
		array_logits_txt = []
		array_embeddings_img = []
		array_embeddings_txt = []
		label_embedding_pos = []

		labels = labels.to(device, non_blocking=True)

		for x, filename_wsi in enumerate(IDs):

			inputs_embedding = None
			encoded_text = None

			diagnosis_wsi = None
			features_np = None
			
			labels_np = labels[x].cpu().data.numpy().flatten()

			prob_pre = np.random.rand(1)[0]

			features_np = utils.get_features(instance_dir, filename_wsi, WEIGHTS)

			diagnosis_wsi = utils.get_diagnosis(filename_wsi, dict_reports)
				
			_, cls_img, _, cls_txt, _ = get_predictions_embeddings(features_np, diagnosis_wsi, labels[x], phase)
			

			array_embeddings_img.append(cls_img)
			array_embeddings_txt.append(cls_txt)
			label_embedding_pos.append(torch.tensor(1))


		array_embeddings_img = torch.stack(array_embeddings_img, dim=0).to(device)
		array_embeddings_txt = torch.stack(array_embeddings_txt, dim=0).to(device)
		label_embedding_pos = torch.stack(label_embedding_pos, dim=0).to(device)

		if (MODALITY is MOD.contrastive):

			loss_representation_cosine = criterion_representation_cosine(array_embeddings_img, array_embeddings_txt, label_embedding_pos)
			representation_loss_cosine = representation_loss_cosine + ((1 / (i+1)) * (loss_representation_cosine.item() - representation_loss_cosine))
			
			loss_representation_rmse = criterion_representation_rmse(array_embeddings_img, array_embeddings_txt) #+ criterion_representation_rmse(cls_txt, cls_img) / 2
			representation_loss_rmse = representation_loss_rmse + ((1 / (i+1)) * (loss_representation_rmse.item() - representation_loss_rmse))

			loss_representation_pairwise = criterion_representation_pairwise(array_embeddings_img, array_embeddings_txt) #+ criterion_representation_rmse(cls_txt, cls_img) / 2
			representation_pairwise = representation_pairwise + ((1 / (i+1)) * (loss_representation_pairwise.item() - representation_pairwise))

			#contrastive loss
			try:
				loss_contrastive = criterion_contrastive(array_embeddings_img, array_embeddings_txt)
				contrastive_loss = contrastive_loss + ((1 / (i+1)) * (loss_contrastive.item() - contrastive_loss))
			except:
				pass
			
			total_loss = contrastive_loss + representation_loss_rmse + representation_loss_cosine

		elif (MODALITY is MOD.contrastive_clip):

			#contrastive loss
			try:
				loss_contrastive = criterion_contrastive(array_embeddings_img, array_embeddings_txt)
				contrastive_loss = contrastive_loss + ((1 / (i+1)) * (loss_contrastive.item() - contrastive_loss))
			except:
				pass
			
			total_loss = contrastive_loss 
		
		print("total_loss: " + str(total_loss))	

		#save loss
		
		print()

	utils.save_loss_function(checkpoint_path, phase, epoch, total_loss)
	return total_loss

def train(epoch):
	
	phase = 'train'
	phase = PHASE[phase]

	total_loss = 0.0

	img_loss = 0.0
	txt_loss = 0.0
	instance_loss = 0.0
	representation_loss_rmse = 0.0
	representation_loss_cosine = 0.0
	contrastive_loss = 0.0
	representation_pairwise = 0.0

	filenames_wsis_i = []
	pred_cancers_i = []
	pred_hgd_i = []
	pred_lgd_i = []
	pred_hyper_i = []
	pred_normal_i = []

	y_pred_i = []
	y_true_i = []

	filenames_wsis_t = []
	pred_cancers_t = []
	pred_hgd_t = []
	pred_lgd_t = []
	pred_hyper_t = []
	pred_normal_t = []

	y_pred_t = []
	y_true_t = []
	
	dataloader_iterator = iter(training_generator_bag)
	iterations = int(len(train_dataset) / BATCH_SIZE_bags)+1

	model.train()
	model.zero_grad(set_to_none=True)
	optimizer.zero_grad(set_to_none=True)

	batch_size_bag = BATCH_SIZE_bags

	for name, param in model.conv_layers.named_parameters():
		#if '10' in name or '11' in name: 
		param.requires_grad = False

	for param in model.bert_model.embeddings.parameters():
		param.requires_grad = False
	"""
	for param in model.domain_predictor.parameters():
		param.requires_grad = True
	"""
	for name, param in model.bert_model.encoder.named_parameters():
		#if '10' in name or '11' in name: 
		if '11' in name: 
			param.requires_grad = True
		else:
			param.requires_grad = False
			
	total_params = sum(p.numel() for p in model.parameters())
	print(f'{total_params:,} total parameters.')
	total_trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad)
	print(f'{total_trainable_params:,} training parameters CNN.')

	for i in range(iterations):
		
		acc_instance_loss = 0.0

		print('[%d], %d / %d ' % (epoch, i, iterations))
		try:
			IDs, labels = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(training_generator_bag)
			IDs, labels = next(dataloader_iterator)
		
		array_logits_img = []
		array_logits_txt = []
		array_embeddings_img = []
		array_embeddings_txt = []
		label_embedding_pos = []

		labels = labels.to(device, non_blocking=True)

		for x, filename_wsi in enumerate(IDs):
			#print(filename_wsi)
			inputs_embedding = None
			encoded_text = None

			diagnosis_wsi = None
			features_np = None
			
			labels_np = labels[x].cpu().data.numpy().flatten()

			prob_pre = np.random.rand(1)[0]

			if (PREPROCESSED_DATA==False and prob_pre>=AUGMENT_PROB_THRESHOLD):

				features_np = compute_features(instance_dir, filename_wsi)

			else:
				
				features_np = utils.get_features(instance_dir, filename_wsi, WEIGHTS)

			diagnosis_wsi = utils.get_diagnosis(filename_wsi, dict_reports)

			prob_pre = np.random.rand(1)[0]
			if (prob_pre>=AUGMENT_PROB_REPORT):
				#print(diagnosis_wsi)
				diagnosis_wsi = utils.get_augment_diagnosis(filename_wsi, translated, diagnosis_wsi)
				#print(diagnosis_wsi)
				
			_, cls_img, _, cls_txt, _ = get_predictions_embeddings(features_np, diagnosis_wsi, labels[x], phase)
			

			array_embeddings_img.append(cls_img)
			array_embeddings_txt.append(cls_txt)
			label_embedding_pos.append(torch.tensor(1))


		array_embeddings_img = torch.stack(array_embeddings_img, dim=0).to(device)
		array_embeddings_txt = torch.stack(array_embeddings_txt, dim=0).to(device)
		label_embedding_pos = torch.stack(label_embedding_pos, dim=0).to(device)

		if (MODALITY is MOD.contrastive):

			loss_representation_cosine = criterion_representation_cosine(array_embeddings_img, array_embeddings_txt, label_embedding_pos)
			representation_loss_cosine = representation_loss_cosine + ((1 / (i+1)) * (loss_representation_cosine.item() - representation_loss_cosine))
			
			loss_representation_rmse = criterion_representation_rmse(array_embeddings_img, array_embeddings_txt) #+ criterion_representation_rmse(cls_txt, cls_img) / 2
			representation_loss_rmse = representation_loss_rmse + ((1 / (i+1)) * (loss_representation_rmse.item() - representation_loss_rmse))

			loss_representation_pairwise = criterion_representation_pairwise(array_embeddings_img, array_embeddings_txt) #+ criterion_representation_rmse(cls_txt, cls_img) / 2
			representation_pairwise = representation_pairwise + ((1 / (i+1)) * (loss_representation_pairwise.item() - representation_pairwise))

			#contrastive loss
			try:
				loss_contrastive = criterion_contrastive(array_embeddings_img, array_embeddings_txt)
				contrastive_loss = contrastive_loss + ((1 / (i+1)) * (loss_contrastive.item() - contrastive_loss))
			except:
				pass

			loss = loss_contrastive + loss_representation_cosine + loss_representation_rmse #+ loss_representation_pairwise

		elif (MODALITY is MOD.contrastive_clip):

			loss_contrastive = criterion_contrastive(array_embeddings_img, array_embeddings_txt)
			contrastive_loss = contrastive_loss + ((1 / (i+1)) * (loss_contrastive.item() - contrastive_loss))

			loss = loss_contrastive

		loss.backward()
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)
		model.zero_grad(set_to_none=True)

		#total_loss = total_loss + ((1 / (i+1)) * (loss.item() - total_loss))

		total_loss = total_loss + ((1 / (i+1)) * (loss.item() - total_loss))
		

		if (MODALITY is MOD.contrastive):
			
			print("representation_loss_cosine: " + str(representation_loss_cosine))
			print("representation_loss_rmse: " + str(representation_loss_rmse))
			print("contrastive_loss: " + str(contrastive_loss))

		elif (MODALITY is MOD.contrastive_clip):
			
			print("contrastive_loss: " + str(contrastive_loss))

		print()

	#save loss
	utils.save_loss_function(checkpoint_path, phase, epoch, total_loss)
	return total_loss

epoch = 0

best_loss = 100000.0

EARLY_STOP_NUM = 10
early_stop_cont = 0
epoch = 0

batch_size_instance = int(BATCH_SIZE_str)

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
		
	#train
	train_loss = train(epoch)

	valid_loss = evaluate(epoch)
	
	if (best_loss>valid_loss):
		early_stop_cont = 0
		print ("=> Saving a new best model")
		print("previous loss : " + str(best_loss) + ", new loss function: " + str(valid_loss))
		best_loss = valid_loss

		try:
			torch.save(model.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
		except:
			try:
				torch.save(model.state_dict(), model_weights_filename)
			except:
				torch.save(model, model_weights_filename)
		
	else:
		early_stop_cont = early_stop_cont+1
	
	#save hyper
	utils.save_hyperparameters(checkpoint_path, N_CLASSES, EMBEDDING_bool, lr)
	
	epoch = epoch + 1

torch.cuda.empty_cache()
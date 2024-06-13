import numpy as np 
import torch
import torch.nn.functional as F
import torch.utils.data

from glob import glob
from matplotlib.pyplot import imshow
from numba import jit
from nystrom_attention import NystromAttention
from transformers import BertModel
from utils import MOD, ALG, PHASE

###### features from CNN backbone
class CNN_Encoder(torch.nn.Module):
	def __init__(self, CNN_TO_USE):

		super(CNN_Encoder, self).__init__()
		
		#CNN_TO_USE = 'resnet34'

		pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)
		if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.fc.in_features
		elif (('densenet' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.classifier.in_features
		elif ('mobilenet' in CNN_TO_USE):
			fc_input_features = pre_trained_network.classifier[1].in_features
	
		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
		self.fc_feat_in = fc_input_features

	def forward(self, x):

		conv_layers_out=self.conv_layers(x)

		features = conv_layers_out.view(-1, self.fc_feat_in)

		return features

class ProjectionHead(torch.nn.Module):
	def __init__(self, embedding_dim: int = 128, projection_dim: int = 128, dropout: float = 0.1):
		super().__init__()

		self.projection = torch.nn.Linear(embedding_dim, projection_dim)
		self.gelu = torch.nn.GELU()
		self.fc = torch.nn.Linear(projection_dim, projection_dim)

		self.dropout = torch.nn.Dropout(dropout)
		self.layer_norm = torch.nn.LayerNorm(projection_dim)


	def forward(self, x):
		projected = self.projection(x)
		x = self.gelu(projected)
		x = self.fc(x)
		x = self.dropout(x)

		x += projected

		return self.layer_norm(x)

########## additive MIL
class Additive_MIL(torch.nn.Module):
	def __init__(self, N_CLASSES = 5, fc_input_features = 512, hidden_space_len=128, TEMPERATURE = 0.07):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(Additive_MIL, self).__init__()

		#CNN_TO_USE = 'resnet34'

		self.N_CLASSES = N_CLASSES
		self.TEMPERATURE = TEMPERATURE

		self.fc_feat_in = fc_input_features

		#number elements first embedding layer (from output resnet to first FCN)
		self.E = hidden_space_len
		#number elements input attention pooler
		self.L = self.E
		#number elements output attention pooler
		self.D = hidden_space_len
		#number channels output attention pooler
		self.K = N_CLASSES

		#######general components

		self.embedding_feature_img = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
		self.post_embedding_feature_img = torch.nn.Linear(in_features=self.E, out_features=self.E)

		#attention network
		self.attention = torch.nn.Sequential(
			torch.nn.Linear(self.L, self.D),
			torch.nn.Tanh(),
			torch.nn.Linear(self.D, self.K)
		)

		#from cnn embedding to intermediate embedding
		

	def forward(self, features):		
		A = None

		embedding_layer_img = self.embedding_feature_img(features)

		A = self.attention(embedding_layer_img)

		A = torch.transpose(A, 1, 0)
		#A = F.softmax(A , dim=1)
		A = F.softmax(A / self.TEMPERATURE, dim=1)

		wsi_embedding = torch.mm(A, embedding_layer_img)
		
		return wsi_embedding

########## additive MIL
class AB_MIL(torch.nn.Module):
	def __init__(self, N_CLASSES = 5, fc_input_features = 512, hidden_space_len=128, TEMPERATURE = 0.07):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(AB_MIL, self).__init__()

		#CNN_TO_USE = 'resnet34'

		self.N_CLASSES = N_CLASSES
		self.TEMPERATURE = TEMPERATURE

		self.fc_feat_in = fc_input_features

		#number elements first embedding layer (from output resnet to first FCN)
		self.E = hidden_space_len
		#number elements input attention pooler
		self.L = self.E
		#number elements output attention pooler
		self.D = hidden_space_len
		#number channels output attention pooler
		self.K = N_CLASSES

		#######general components

		self.embedding_feature_img = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
		self.post_embedding_feature_img = torch.nn.Linear(in_features=self.E, out_features=self.E)

		#attention network
		self.attention = torch.nn.Sequential(
			torch.nn.Linear(self.L, self.D),
			torch.nn.Tanh(),
			torch.nn.Linear(self.D, 1)
		)

		#from cnn embedding to intermediate embedding
		

	def forward(self, features):		
		A = None

		embedding_layer_img = self.embedding_feature_img(features)

		A = self.attention(embedding_layer_img)

		A = torch.transpose(A, 1, 0)
		#A = F.softmax(A , dim=1)
		A = F.softmax(A / self.TEMPERATURE, dim=1)

		wsi_embedding = torch.mm(A, embedding_layer_img)
		
		return wsi_embedding


################ CLAM based

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, torch.nn.Linear):
			torch.nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, torch.nn.BatchNorm1d):
			torch.nn.init.constant_(m.weight, 1)
			torch.nn.init.constant_(m.bias, 0)

class Attn_Net(torch.nn.Module):

	def __init__(self, L = 128, D = 128, dropout = False, n_classes = 1):
		super(Attn_Net, self).__init__()
		self.module = [
			torch.nn.Linear(L, D),
			torch.nn.Tanh()]

		if dropout:
			self.module.append(torch.nn.Dropout(0.25))

		self.module.append(torch.nn.Linear(D, n_classes))
		
		self.module = torch.nn.Sequential(*self.module)
	
	def forward(self, x):

		return self.module(x) # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
	L: input feature dimension
	D: hidden layer dimension
	dropout: whether to use dropout (p = 0.25)
	n_classes: number of classes 
"""
class Attn_Net_Gated(torch.nn.Module):
	def __init__(self, L = 128, D = 128, dropout = False, n_classes = 1):
		super(Attn_Net_Gated, self).__init__()
		self.attention_a = [
			torch.nn.Linear(L, D),
			torch.nn.Tanh()]
		
		self.attention_b = [torch.nn.Linear(L, D),
							torch.nn.Sigmoid()]
		if dropout:
			self.attention_a.append(torch.nn.Dropout(0.25))
			self.attention_b.append(torch.nn.Dropout(0.25))

		self.attention_a = torch.nn.Sequential(*self.attention_a)
		self.attention_b = torch.nn.Sequential(*self.attention_b)
		
		self.attention_c = torch.nn.Linear(D, n_classes)

	def forward(self, x):
		a = self.attention_a(x)
		b = self.attention_b(x)
		A = a.mul(b)
		A = self.attention_c(A)  # N x n_classes
		return A, x		

class CLAM_SB(torch.nn.Module):
	def __init__(self, gate = False, size_arg = "small", dropout = True, k_sample=8, n_classes=2,
		instance_loss_fn=torch.nn.CrossEntropyLoss(), subtyping=False, device = None):
		
		super(CLAM_SB, self).__init__()
		#self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
		self.size_dict = {"small": [512, 128, 128], "big": [512, 128, 128]}
		self.device = device
		size = self.size_dict[size_arg]
		
		self.fc = torch.nn.Linear(size[0], size[1])
		self.relu = torch.nn.ReLU()

		fc = [torch.nn.Linear(size[0], size[1]), torch.nn.ReLU()]
		
		self.dropout = torch.nn.Dropout(p=0.2)

		self.attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
		
		instance_classifiers = [torch.nn.Linear(size[1], 2) for i in range(n_classes)]
		
		self.instance_classifiers = torch.nn.ModuleList(instance_classifiers)
		self.k_sample = k_sample

		if (instance_loss_fn == 'svm'):
			self.instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda()
		else:
			self.instance_loss_fn = torch.nn.CrossEntropyLoss()

		self.n_classes = n_classes
		self.subtyping = subtyping

		initialize_weights(self)

	def relocate(self):
		device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.attention_net = self.attention_net.to(device)
		self.instance_classifiers = self.instance_classifiers.to(device)
	
	@staticmethod
	def create_positive_targets(length, device):
		return torch.full((length, ), 1, device=device).long()
	@staticmethod
	def create_negative_targets(length, device):
		return torch.full((length, ), 0, device=device).long()
	
	#instance-level evaluation for in-the-class attention branch
	def inst_eval(self, A, h, classifier): 

		if len(A.shape) == 1:
			A = A.view(1, -1)
		top_p_ids = torch.topk(A, self.k_sample)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
		top_n = torch.index_select(h, dim=0, index=top_n_ids)
		p_targets = self.create_positive_targets(self.k_sample, self.device)
		n_targets = self.create_negative_targets(self.k_sample, self.device)

		all_targets = torch.cat([p_targets, n_targets], dim=0)
		all_instances = torch.cat([top_p, top_n], dim=0)

		logits = classifier(all_instances)
		all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, all_targets)

		return instance_loss, all_preds, all_targets
	
	#instance-level evaluation for out-of-the-class attention branch
	def inst_eval_out(self, A, h, classifier):

		if len(A.shape) == 1:
			A = A.view(1, -1)
		top_p_ids = torch.topk(A, self.k_sample)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		p_targets = self.create_negative_targets(self.k_sample, self.device)
		logits = classifier(top_p)
		p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, p_targets)
		return instance_loss, p_preds, p_targets

	def forward(self, features, label=None, instance_eval=True, return_features=False, attention_only=False):

		h = self.fc(features)
		#h = self.relu(h)
		h = self.dropout(h)

		A = self.attention_net(h)  # NxK        
		A = torch.transpose(A, 1, 0)  # KxN
		A = F.softmax(A, dim=1)  # softmax over N

		total_inst_loss = 0.0

		try:
			if instance_eval:
				total_inst_loss = 0.0
				all_preds = []
				all_targets = []

				#inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
				#inst_labels = label

				for i in range(len(self.instance_classifiers)):
					instance_loss = 0.0
					classifier = self.instance_classifiers[i]

					if (label[i]==1):

						instance_loss, preds, targets = self.inst_eval(A, h, classifier)
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						if self.subtyping:

							instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
							all_preds.extend(preds.cpu().numpy())
							all_targets.extend(targets.cpu().numpy())

					total_inst_loss += instance_loss

				if self.subtyping:
					total_inst_loss /= len(self.instance_classifiers)

		except:
			total_inst_loss = 0.0
				
		M = torch.mm(A, h) 
		
		return M, total_inst_loss

class CLAM_MB(CLAM_SB):

	def __init__(self, gate = False, size_arg = "small", dropout = True, k_sample=8, n_classes=4,
		instance_loss_fn = torch.nn.CrossEntropyLoss(), subtyping=False, device = None):

		torch.nn.Module.__init__(self)
		self.device = device

		#self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
		self.size_dict = {"small": [512, 128, 128], "big": [1024, 128, 128]}

		size = self.size_dict[size_arg]

		self.fc = torch.nn.Linear(size[0], size[1])
		self.relu = torch.nn.ReLU()
		self.dropout = torch.nn.Dropout(p=0.2)

		self.attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
		
		#instance_classifier 
		instance_classifiers = [torch.nn.Linear(size[1], 2) for i in range(n_classes)]

		self.instance_classifiers = torch.nn.ModuleList(instance_classifiers)
		self.k_sample = k_sample

		if (instance_loss_fn == 'svm'):
			self.instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda()
		else:
			self.instance_loss_fn = torch.nn.CrossEntropyLoss()

		self.n_classes = n_classes
		self.subtyping = subtyping
		initialize_weights(self)

	def forward(self, features, label, instance_eval=True, return_features=False, attention_only=False):

		h = self.fc(features)
		#h = self.relu(h)
		h = self.dropout(h)

		A = self.attention_net(h)  # NxK        
		A = torch.transpose(A, 1, 0)  # KxN
		A = F.softmax(A, dim=1)  # softmax over N

		total_inst_loss = 0.0

		try:
			if instance_eval:
				total_inst_loss = 0.0
				all_preds = []
				all_targets = []

				#inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
				#inst_labels = label

				for i in range(len(self.instance_classifiers)):
					instance_loss = 0.0
					classifier = self.instance_classifiers[i]

					if (label[i]==1):

						instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						if self.subtyping:

							instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
							all_preds.extend(preds.cpu().numpy())
							all_targets.extend(targets.cpu().numpy())

					total_inst_loss += instance_loss

				if self.subtyping:
					total_inst_loss /= len(self.instance_classifiers)

		except:
			total_inst_loss = 0.0

		#logits pgp
		M = torch.mm(A, h) 
		
		return M, total_inst_loss

########## DSMIL
class IClassifier(torch.nn.Module):
	def __init__(self, input_size, feature_size, output_class):
		super(IClassifier, self).__init__()

		self.embedding = torch.nn.Linear(input_size, feature_size)
		self.fc = torch.nn.Linear(feature_size, output_class)
		
	def forward(self, x):
		
		feats = self.embedding(x)
		c = self.fc(feats) # N x C
		return feats.view(feats.shape[0], -1), c

class BClassifier(torch.nn.Module):

	def __init__(self, input_size, output_class, device, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
		super(BClassifier, self).__init__()
		if nonlinear:
			self.q = torch.nn.Sequential(torch.nn.Linear(input_size, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.Tanh())
		else:
			self.q = torch.nn.Linear(input_size, 128)
		if passing_v:
			self.v = torch.nn.Sequential(
				torch.nn.Dropout(dropout_v),
				torch.nn.Linear(input_size, input_size),
				torch.nn.ReLU()
			)
		else:
			self.v = torch.nn.Identity()

		### 1D convolutional layer that can handle multiple class (including binary)
		self.fcc = torch.nn.Conv1d(output_class, output_class, kernel_size=input_size)  

		self.device = device
		self.output_class = output_class

	def forward(self, feats, c): # N x K, N x C

		V = self.v(feats) # N x V, unsorted
		Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted

		# handle multiple classes without for loop
		_, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
		m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
		q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
		A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
		A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=self.device)), 0) # normalize attention scores, A in shape N x C, 
		wsi_embedding = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
		
		return wsi_embedding#, A, B 

class DSMIL(torch.nn.Module):
	def __init__(self, fc_input_features, hidden_space_len, N_CLASSES, device):
		super(DSMIL, self).__init__()
		#self.i_classifier = i_classifier
		#self.b_classifier = b_classifier
		self.device = device

		self.input_features = fc_input_features
		self.E = hidden_space_len
		self.K = N_CLASSES

		self.i_classifier = IClassifier(self.input_features, self.E, N_CLASSES)
		
		self.b_classifier = BClassifier(self.E, N_CLASSES, device = self.device)
		

	def forward(self, x):

		feats, classes = self.i_classifier(x)

		#prediction_bag, A, B = self.b_classifier(feats, classes)
		features = self.b_classifier(feats, classes)

		#return classes, prediction_bag, A, B	
		return features

######### transMIL
class TransLayer(torch.nn.Module):

	def __init__(self, norm_layer=torch.nn.LayerNorm, dim=128):
		super().__init__()
		self.norm = norm_layer(dim)
		self.attn = NystromAttention(
			dim = dim,
			dim_head = dim//8,
			heads = 8,
			num_landmarks = dim//2,    # number of landmarks
			pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
			residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
			dropout=0.1
		)

	def forward(self, x):
		x = x + self.attn(self.norm(x))

		return x


class PPEG(torch.nn.Module):
	def __init__(self, dim=128):
		super(PPEG, self).__init__()
		self.proj = torch.nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
		self.proj1 = torch.nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
		self.proj2 = torch.nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

	def forward(self, x, H, W):
		B, _, C = x.shape
		cls_token, feat_token = x[:, 0], x[:, 1:]
		cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
		x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
		x = x.flatten(2).transpose(1, 2)
		x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
		return x


class TransMIL(torch.nn.Module):
	def __init__(self, fc_input_features, hidden_space_len, N_CLASSES):
		super(TransMIL, self).__init__()

		self.fc_feat_in = fc_input_features
		self.E = 128

		self.n_classes = N_CLASSES
		#pgp
		self.pos_layer = PPEG(dim=self.E)
		#self._fc1 = torch.nn.Sequential(torch.nn.Linear(self.fc_feat_in, self.E), torch.nn.ReLU())
		self._fc1 = torch.nn.Sequential(torch.nn.Linear(self.fc_feat_in, self.E))
		
		self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.E))

		self.layer1 = TransLayer(dim=self.E)
		self.layer2 = TransLayer(dim=self.E)
		self.norm = torch.nn.LayerNorm(self.E)
		self._fc2 = torch.nn.Linear(self.E, self.n_classes)



	def forward(self, features):

		features = torch.reshape(features, (1, features.shape[0], features.shape[1]))

		#pgp
		h = self._fc1(features) #[B, n, 512]

		#---->pad
		H = h.shape[1]
		_H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
		add_length = _H * _W - H

		h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

		#---->cls_token
		B = h.shape[0]
		cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
		h = torch.cat((cls_tokens, h), dim=1)

		#---->Translayer x1
		h = self.layer1(h) #[B, N, 512]

		#---->PPEG
		h = self.pos_layer(h, _H, _W) #[B, N, 512]

		#---->Translayer x2
		h = self.layer2(h) #[B, N, 512]

		#---->cls_token
		#wsi_embedding = self.norm(h)[:,0]
		
		wsi_embedding = h[:,0]

		return wsi_embedding

class MultimodalArchitecture(torch.nn.Module):
	def __init__(self, device, CNN_TO_USE = 'resnet34', MODALITY = 'multimodal_clip', MIL_architecture = 'ADMIL', N_CLASSES = 5, fc_input_features = 512, hidden_space_len=128, TEMPERATURE = 0.07):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(MultimodalArchitecture, self).__init__()

		#common parameters
		self.TEMPERATURE = TEMPERATURE
		self.device = device
		self.tanh = torch.nn.Tanh()
		self.relu = torch.nn.ReLU()
		self.activation = self.tanh
		self.hidden_space_len = hidden_space_len
		self.N_CLASSES = N_CLASSES
		self.O = self.hidden_space_len
		
		self.MODALITY = MODALITY
		self.MODALITY = MOD[MODALITY]

		#image encoder
		self.conv_layers = CNN_Encoder(CNN_TO_USE)

		#output from resnet34
		self.fc_feat_in = fc_input_features

		self.MIL_architecture = MIL_architecture
		self.MIL_architecture = ALG[MIL_architecture]

		self.dropout_img = torch.nn.Dropout(p=0.1)
		
		instance_loss_fn = 'svm' 
		instance_loss_fn = 'ce' 

		if (self.MIL_architecture is ALG.CLAM_MB):
			self.MIL_algorithm = CLAM_MB(n_classes = self.N_CLASSES, instance_loss_fn = instance_loss_fn, device = self.device)

		elif (self.MIL_architecture is ALG.CLAM_SB):
			self.MIL_algorithm = CLAM_SB(n_classes = self.N_CLASSES, instance_loss_fn = instance_loss_fn, device = self.device)

		elif (self.MIL_architecture is ALG.DSMIL):
			self.MIL_algorithm = DSMIL(fc_input_features = self.fc_feat_in , hidden_space_len = self.hidden_space_len, N_CLASSES = self.N_CLASSES, device = self.device)

		elif (self.MIL_architecture is ALG.transMIL):
			self.MIL_algorithm = TransMIL(fc_input_features = self.fc_feat_in , hidden_space_len = self.hidden_space_len, N_CLASSES = self.N_CLASSES)

		elif (self.MIL_architecture is ALG.ADMIL):
			self.MIL_algorithm = Additive_MIL(N_CLASSES = N_CLASSES, fc_input_features = self.fc_feat_in, hidden_space_len=self.hidden_space_len)

		elif (self.MIL_architecture is ALG.ABMIL):
			self.MIL_algorithm = AB_MIL(N_CLASSES = N_CLASSES, fc_input_features = self.fc_feat_in, hidden_space_len=self.hidden_space_len)

		self.L = self.hidden_space_len
		self.D = self.hidden_space_len

		if ((self.MIL_architecture is ALG.DSMIL or self.MIL_architecture is ALG.CLAM_MB) or self.MIL_architecture is ALG.ADMIL):
			self.attention_channel = torch.nn.Sequential(
					torch.nn.Linear(self.L, self.D),
					torch.nn.Tanh(),
					torch.nn.Linear(self.D, 1)
				)
		

		# parameters bert

		clinical_bert_token_size = 768
		
		#########text parameters		
		self.clinical_bert_token_size = clinical_bert_token_size
		#img projection
		self.LayerNorm = torch.nn.LayerNorm(self.clinical_bert_token_size, eps=1e-5)
		self.dropout_bert = torch.nn.Dropout(0.1)

		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			
			#self.pooler = None
			self.bert_model = BertModel.from_pretrained(bert_chosen, 
														output_attentions=True, 
														output_hidden_states=True)
		except:
			bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
			#self.pooler = None
			self.bert_model = BertModel.from_pretrained(bert_chosen, 
														output_attentions=True, 
														output_hidden_states=True)
														
		#from CLS_TOKEN to common embedding (like for images)
		self.embedding_output_txt = torch.nn.Linear(in_features=self.clinical_bert_token_size, out_features=self.O)		

		#COMMON LAYERS MULTIMODAL
		#output of first common layer
		self.I = self.hidden_space_len

		self.projection_img = ProjectionHead(embedding_dim = 128, projection_dim = 128)
		self.projection_txt = ProjectionHead(embedding_dim = 128, projection_dim = 128)

		if (self.MIL_architecture is ALG.DSMIL):
			self.classifier = torch.nn.Conv1d(1, self.N_CLASSES, kernel_size=self.I)

		else:
			self.classifier = torch.nn.Linear(in_features=self.I, out_features=self.N_CLASSES)


		#self.domain_predictor = domain_predictor()

	def get_txt_token(self, input_txt, start_pos = 0):

		input_ids_txt = input_txt["input_ids"]
		token_type_txt = input_txt["token_type_ids"] #sum +1
		attention_mask = input_txt["attention_mask"]

		bsz = 1
		seq_length = len(input_ids_txt)
		sentence_val = 0

		if (start_pos > 0):

			input_ids_txt = input_ids_txt[1:]
			attention_mask = attention_mask[1:]
			seq_length = seq_length - 1
			#sentence_val = 1

		#attention_mask = torch.zeros(bsz, seq_length).long().to(device)
		#attention_mask = torch.ones(bsz, seq_length).long().to(device)

		attention_mask = torch.as_tensor([attention_mask]).long().to(self.device)
		#print(attention_mask)
		#word_embed
		input_ids_txt = torch.as_tensor([input_ids_txt]).long().to(self.device)
		txt_token_embeds = self.bert_model.embeddings.word_embeddings(input_ids_txt)

		#txt_tok (0 if unimodal, 1 otherwise)
		txt_tok = (
			torch.LongTensor(bsz, seq_length)
			.fill_(sentence_val)
			.to(self.device)
		)

		#position
		end_pos = start_pos + seq_length
		position_ids = torch.arange(start_pos, end_pos)
		position_ids = position_ids.type(torch.LongTensor).to(self.device)

		position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)

		#sum
		position_embeddings = self.bert_model.embeddings.position_embeddings(position_ids)
		token_type_embeddings = self.bert_model.embeddings.token_type_embeddings(txt_tok)

		#embeddings_txt = txt_token_embeds + position_embeddings #+ token_type_embeddings
		embeddings_txt = txt_token_embeds + token_type_embeddings + position_embeddings 

		embeddings_txt = self.bert_model.embeddings.LayerNorm(embeddings_txt)
		embeddings_txt = self.dropout_bert(embeddings_txt)

		return embeddings_txt, attention_mask

	def forward(self, input_img, input_txt, labels, phase):
		img_prob = None
		intermediate_embedding_img_1 = None
		txt_prob = None
		intermediate_embedding_txt_1 = None
		total_inst_loss = None

		phase = PHASE[phase]

		####process images (features pre-processed)
		if input_img is not None:
			
			#dk = max(1,np.log10(self.D))
		
			CLAM_flag = True if phase is PHASE.train else False

			if (self.MIL_architecture is ALG.CLAM_MB or self.MIL_architecture is ALG.CLAM_SB):
				wsi_embedding, total_inst_loss = self.MIL_algorithm(input_img, label = labels, instance_eval = CLAM_flag)
			else:
				wsi_embedding = self.MIL_algorithm(input_img)

			if ((self.MIL_architecture is ALG.DSMIL or self.MIL_architecture is ALG.CLAM_MB) or self.MIL_architecture is ALG.ADMIL):

				attention_channel = self.attention_channel(wsi_embedding)
				attention_channel = torch.transpose(attention_channel, 1, 0)
				attention_channel = F.softmax(attention_channel / self.TEMPERATURE, dim=1)
				#attention_channel = F.softmax(attention_channel, dim=1)
				output_img = torch.mm(attention_channel, wsi_embedding)

			else:
				output_img = wsi_embedding
			
			output_img = self.projection_img(output_img)

			#output_img = self.activation(output_img)
			output_img = output_img.view(-1)

			intermediate_embedding_img_1 = output_img

			if (self.MIL_architecture is ALG.DSMIL):
				output_img = output_img.unsqueeze(0).unsqueeze(0)
				img_prob = self.classifier(output_img)
				img_prob = img_prob.squeeze(2).view(-1)

			else:
				img_prob = self.classifier(output_img).view(-1)


		if input_txt is not None:

			#txt embedding
			txt_embedding, attention_mask_txt = self.get_txt_token(input_txt)

			#txt
			outputs_txt = self.bert_model.encoder(txt_embedding, attention_mask_txt)

			pooled_output_txt = self.bert_model.pooler(outputs_txt[-1])

			pooled_output_txt = self.dropout_bert(pooled_output_txt)
			intermediate_txt = self.embedding_output_txt(pooled_output_txt)

			output_txt = intermediate_txt

			output_txt = self.projection_txt(intermediate_txt)
			#output_txt = self.activation(output_txt)
			output_txt = output_txt.view(-1)
			intermediate_embedding_txt_1 = output_txt
 
			#output_txt = self.dropout_bert(output_txt)
			if (self.MIL_architecture is ALG.DSMIL):
				output_txt = output_txt.unsqueeze(0).unsqueeze(0)
				txt_prob = self.classifier(output_txt)
				txt_prob = txt_prob.squeeze(2).view(-1)
			else:
				txt_prob = self.classifier(output_txt).view(-1)

			

		return img_prob, intermediate_embedding_img_1, txt_prob, intermediate_embedding_txt_1, total_inst_loss

if __name__ == "__main__":
	pass
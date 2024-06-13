import torch
from torch.utils import data
import numpy as np 
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from enum_multi import PHASE

#sampler
class Balanced_Multimodal(data.sampler.Sampler):

	def __init__(self, dataset, indices=None, num_samples=None, alpha = 0.5):

		self.indices = list(range(len(dataset)))             if indices is None else indices

		self.num_samples = len(self.indices)             if num_samples is None else num_samples

		class_sample_count = [0,0,0,0,0]


		class_sample_count = np.sum(dataset[:,1:],axis=0)

		min_class = np.argmin(class_sample_count)
		class_sample_count = np.array(class_sample_count)
		weights = []
		for c in class_sample_count:
			weights.append((c/class_sample_count[min_class]))

		ratio = np.array(weights).astype(np.float)

		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			for l in label:
				if l in label_to_count:
					label_to_count[l] += 1
				else:
					label_to_count[l] = 1

		weights = []

		for idx in self.indices:
			c = 0
			for j, l in enumerate(self._get_label(dataset, idx)):
				c = c+(1/label_to_count[l])#*ratio[l]

			weights.append(c/(j+1))
			#weights.append(c)
			
		self.weights_original = torch.DoubleTensor(weights)

		self.weights_uniform = np.repeat(1/self.num_samples, self.num_samples)

		#print(self.weights_a, self.weights_b)

		beta = 1 - alpha
		self.weights = (alpha * self.weights_original) + (beta * self.weights_uniform)


	def _get_label(self, dataset, idx):
		labels = np.where(dataset[idx,1:]==1)[0]
		#print(labels)
		#labels = dataset[idx,2]
		return labels

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

class ImbalancedDatasetSampler_single_label(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
			
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
				
		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
				   for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset[idx,1]
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

#dataloaders
#data loader at patch-level
class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, mode, pipeline):
		self.list_IDs = list_IDs
		self.pipeline_transform = pipeline
		self.mode = mode

		
		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index][0]
		# Load data and get label
		"""
		with open(ID, 'rb') as fin:
			X = pyspng.load(fin.read())
		"""
		X = Image.open(ID)
		X = np.asarray(X)
		#img.close()

		if (self.mode is PHASE.train):
			X = self.pipeline_transform(image=X)['image']

		#data transformation
		input_tensor = self.preprocess(X).type(torch.FloatTensor)

		#return input_tensor
		return input_tensor

#data loader at WSI-level
class Dataset_bag_multilabel(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		
		# Load data and get label
		
		y = self.labels[index]
		y = torch.as_tensor(y.tolist() , dtype=torch.float32)
		
		return ID, y

class Dataset_bag_multiclass(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		
		# Load data and get label
		
		y = self.labels[index]
		y = np.argmax(y)

		y = torch.as_tensor(y, dtype=torch.long)

		return ID, y

if __name__ == "__main__":
	pass
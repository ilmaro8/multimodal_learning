from enum import Enum

class MOD(Enum):
	multimodal = 0
	unimodal = 1
	contrastive = 2
	multimodal_clip = 3
	contrastive_clip = 4

class ALG(Enum):
	ADMIL = 0
	transMIL = 1
	CLAM_SB = 2
	CLAM_MB = 3
	DSMIL = 4
	ABMIL = 5

class PHASE(Enum):
	train = 0
	valid = 1
	test = 2

class TYPE(Enum):
	images = 0
	reports = 1

class AGGREGATION(Enum):
	normal = 0
	micro = 1

class METRIC(Enum):
	precision_at = 0
	mAP = 1

if __name__ == "__main__":
	pass
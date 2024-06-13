import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

	def __init__(self):
		super(ContrastiveLoss, self).__init__()
		self.eps=1e-6

	def forward(self, a, b):

		pdist = torch.nn.PairwiseDistance(2)
		loss_contrastive = pdist(a,b)
		loss_contrastive = torch.mean(loss_contrastive)

		return loss_contrastive

class RMSELoss(torch.nn.Module):
	def __init__(self, eps=1e-6):
		super().__init__()
		self.mse = torch.nn.MSELoss()
		self.eps = eps
		
	def forward(self,yhat,y):
		loss = torch.sqrt(self.mse(yhat,y) + self.eps)
		return loss

class CosineDistance(torch.nn.Module):
	

	def __init__(self, dim: int = 1, eps: float = 1e-8):
		super(CosineDistance, self).__init__()
		self.dim = dim
		self.eps = eps

	def forward(self, x1, x2):
		return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)

class NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1

		self.mask = self.mask_correlated_samples(batch_size, self.world_size)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size, world_size):
		N = 2 * batch_size * world_size
		mask = torch.ones((N, N), dtype=bool)
		mask = mask.fill_diagonal_(0)
		for i in range(batch_size * world_size):
			mask[i, batch_size * world_size + i] = 0
			mask[batch_size * world_size + i, i] = 0
		return mask

	def forward(self, z_i, z_j):
		"""
		We do not sample negative examples explicitly.
		Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
		"""
		N = 2 * self.batch_size * self.world_size

		z = torch.cat((z_i, z_j), dim=0)
		if self.world_size > 1:
			z = torch.cat(GatherLayer.apply(z), dim=0)

		sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

		sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
		sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

		# We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
		positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
		negative_samples = sim[self.mask].reshape(N, -1)

		labels = torch.zeros(N).to(positive_samples.device).long()
		logits = torch.cat((positive_samples, negative_samples), dim=1)
		loss = self.criterion(logits, labels)
		loss /= N
		return loss


class CLIP_Loss(torch.nn.Module):
    def __init__(self, batch_size = 4, temperature = 0.07):
        super(CLIP_Loss, self).__init__()
        
        self.batch_size = batch_size
        self.temperature = temperature
        
    def forward(self,image_embeddings,text_embeddings):

        similarity_matrix = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings.unsqueeze(0), dim=-1)
    
        diagonal = similarity_matrix.diag().view(-1, 1)
        positive_pairs = diagonal.expand_as(similarity_matrix)
        loss = 0.5 * (1 - positive_pairs) + 0.5 * torch.clamp(similarity_matrix - 0.1, min=0.0)
        loss = loss.sum() / similarity_matrix.size(0)  # Normalize by the batch size
        return loss

if __name__ == "__main__":
	pass
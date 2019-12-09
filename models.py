import torch 
from torch.nn import functional as F 


class LinearRegression(torch.nn.Module):
	def __init__(self):
		super(LinearRegression, self).__init__()
		self.linear = torch.nn.Linear(15,10)

	def forward(self,x):
		y_pred = self.linear(x)
		return y_pred


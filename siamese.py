import torch
import numpy as np
from torch import nn

class SiameseNetworks(nn.Module):
	def __init__(self):
		super(SiameseNetworks,self).__init__()
		self.cnn_1=nn.Conv2d(1,64,kernel_size=10)
		self.cnn_2=nn.Conv2d(64,128,kernel_size=7)
		self.cnn_3=nn.Conv2d(128,128,kernel_size=4)
		self.cnn_4=nn.Conv2d(128,256,kernel_size=4)
	def forward_once(self,x):
		#forward pass to siamese twins
		self.block_1=nn.maxpool2d(nn.relu(self.cnn_1(x)),kernel_size=2)
		self.block_2=nn.maxpool2d(nn.relu(self.cnn_2(self.block_1)),kernel_size=2)
		self.block_3=nn.maxpool2d(nn.relu(self.cnn_2(self.block_2)),kernel_size=2)
		self.block_4=nn.relu(self.cnn_2(self.block_3))
		self.fc_1=nn.linear(9216,4096)(self.block_4)
		self.fc_2=nn.linear(4096,1)
	def forward(self,x_1,x_2):
		#forward pass for input x_1
		output_1=self.forward_once(self.x_1)
		#forward pass for input x_2
		output_2=self.forward_once(self.x_2)
		#calculating distance
		distance=torch.abs(output_1-output_2)
		out=self.fc_2(distance)
		#sigmoid of dsitance
		out=self.sigmoid(out)
		return out

if __name__=='__main__':
	network=SiameseNetworks()








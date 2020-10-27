import numpy
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import random
import numpy as np
import pickle

class Omniglot_Train(Dataset):
	def __init__(self,path,transform=None):
		super(Omniglot_Train,self).__init__()
		self.transform=transform
		self.data,self.classes=self.load_data(path)

	def load_data(self,path):
		data={}
		idx=0
		for alphabet in os.listdir(path)[1:]:#30 alphabets
			for character in os.listdir(os.path.join(path,alphabet))[1:]:# take  all characters in that alphabet
				 #idx will be the class(each character is a class)
				idx=idx+1
				data[str(idx)]=[]
				for file in os.listdir(os.path.join(path,alphabet,character))[1:13]: # take 12 drawer's image in each character
					file_path=os.path.join(path,alphabet,character,file)
					data[str(idx)].append(Image.open(file_path).convert("L"))
				
		return data,idx

	def __len__(self):
		return 30* self.classes*12

	def __getitem__(self,index):
		img1_idx=str(random.randint(1,self.classes))
		
		if index%2==1:
			label=1.0
			img1=random.choice(self.data[img1_idx])
			img2=random.choice(self.data[img1_idx])
		else:
			#diffferent class
			label=0.0
			img2_idx=img1_idx
			'''iterarate whilel loop till we 
				get different class (may or may not be with in the alphabet) '''
			while (img2_idx==img1_idx): 
				img2_idx=str(random.randint(1,self.classes))
			img1=random.choice(self.data[img1_idx])
			img2=random.choice(self.data[img2_idx])
		if self.transform:
			img1=self.transform(img1)
			img2=self.transform(img2)
		return img1,img2,torch.from_numpy(np.array([label],dtype='float32'))




class Omniglot_Validation(Dataset):
	def __init__(self,path,transform=None):
		super(Omniglot_Test,self).__init__()
		self.transform=transform
		self.data,self.classes=self.load_data(path)

	def load_data(self,path):
		data={}
		idx=0
		for alphabet in os.listdir(path)[1:11]:#10 alphabets
			for character in os.listdir(os.path.join(path,alphabet))[1:]:# take  all characters in that alphabet
				idx=idx+1
				data[str(idx)]=[] #idx will be the class(each character is a class)
				for file in os.listdir(os.path.join(path,alphabet,character))[12:16]: # take 12 drawer's image in each character
					file_path=os.path.join(path,alphabet,character,file)
					data[str(idx)].append(Image.open(file_path).convert("L"))
		return data,idx

	def __len__(self):
		return 10* self.classes*4

	def __getitem__(self,index):
		img1_idx=str(random.randint(0,self.classes))
		
		if index%2==1:
			label=1.0
			img1=str(random.choice(self.data[img1_idx]))
			img2=random.choice(self.data[img1_idx])
		else:
			#diffferent class
			label=0.0
			img2_idx=img1_idx
			'''iterarate whilel loop till we 
				get different class (may or may not be with in the alphabet) '''
			while img2_idx==img1_idx: #
				idx2=random.randint(0,self.classes-1)
			img1=random.choice(self.data[img1_idx])
			img2=random.choice(self.data[img2_idx])
		if self.transform:
			img1=self.transform(img1)
			img2=self.transform(img2)
		return img1,img2,torch.from_numpy(np.array([label],dtype='float32'))


class Omniglot_Test(Dataset):
	def __init__(self,path,transform=None):
		super(Omniglot_Test,self).__init__()
		self.transform=transform
		self.data,self.classes=self.load_data(path)

	def load_data(self,path):
		data={}
		idx=0
		for alphabet in os.listdir(path)[11:]:#10 alphabets
			for character in os.listdir(os.path.join(path,alphabet))[1:]:# take  all characters in that alphabet
				idx=idx+1
				data[str(idx)]=[] #idx will be the class(each character is a class)
				for file in os.listdir(os.path.join(path,alphabet,character))[16:20]: # take 4 drawer's image in each character
					file_path=os.path.join(path,alphabet,character,file)
					data[str(idx)].append(Image.open(file_path).convert("L"))
		return data,idx

	def __len__(self):
		return 10* self.classes*4

	def __getitem__(self,index):
		img1_idx=str(random.randint(0,self.classes))
		
		if index%2==1:
			label=1.0
			img1=str(random.choice(self.data[img1_idx]))
			img2=random.choice(self.data[img1_idx])
		else:
			#diffferent class
			label=0.0
			img2_idx=img1_idx
			'''iterarate whilel loop till we 
				get different class (may or may not be with in the alphabet) '''
			while img2_idx==img1_idx: #
				idx2=random.randint(0,self.classes-1)
			img1=random.choice(self.data[img1_idx])
			img2=random.choice(self.data[img2_idx])
		if self.transform:
			img1=self.transform(img1)
			img2=self.transform(img2)
		return img1,img2,torch.from_numpy(np.array([label],dtype='float32'))






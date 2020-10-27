import mnist
import numpy
import random
from PIL import Image
from torch.utils.data import Dataset,DataLoader

import os
''' Each trial consist of compairision of 1 character image in list-1 with 10 characters in list-2 ,
    so in total  10*10= 200 comparisions in each alphabet. 
'''

class MINSToneshot_trails(Dataset):
    def __int__(self,transform=None):
        np.random.seed(10)
        super(EMINSToneshot_trails, self).__init__()
        self.test_images=self.generate_Oneshot_trails(dataPath,transform=None)
    def generate_Oneshot_trails(self,transform=None):
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()
        images=dict()
        #print(test_images[[11,12],:,:].shape)
        for i in set(test_labels):
            index=[ind for ind,val in enumerate(test_labels) if val==i][:10]
            images[str(i)]=train_images[[index],:,:]
        test_images=dict()
        for i in range(10):
            test_images['aplh'+str(i)]=[]
            test_images['alph'+str(i)]=[images[str(j)][i] for j in range(0,10) ]
        return test_images
    def __len__():
        
        return 10 
    def __getitem__(self,index):
        ''' Index is ignored as we are returning a 2 lists(20 images each)  in one iteration
            For each iteration it will return 2 lists with 10 images each [10 one shot trails]
            Each one-shot trails consist of comparing an image in one list with  10 images in other list '''
        index1=random.randint(0,19)
        index2=random.randint(0,19)

        return self.test_images[index1],self.test_images[index2]



    def MNIST_oneshot_acc(net,dataloader,cuda_check):
        right=0
        error=0
        output=dict()

        for test_img1_batch,test_img2_batch,test_label in dataloader:
            for test_label,test_img1 in enumerate(test_img1_batch,0):
                        for test_img2 in test_img2_batch:

                            if Flags.cuda:
                                test_img1,test_img2=Variable(test_img1.cuda()),Variable(test_img2.cuda())
                            else:
                                test_img1,test_img2=Variable(test_img1),Variable(test_img2)

                            output[str(test_label)].append(net.forward(test_img1,test_img2).data.cpu().numpy())

                        pred=np.argmax(output[str(test_label)])
                        if pred==test_label:
                            right+=1
                        else:
                            error+=1
        return right/(right+error)

if __name__=='main':
    MINSToneshot_trails(Dataset)





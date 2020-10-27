import numpy
import random
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import mnist
from torch.autograd import Variable
import os

''' Each trial consist of compairision of 1 character in drawer-1 with 20 characters in drawer-2 ,
    so in total  20*20= 400 comparisions in each alphabet. 
    This is repeated twice which gives 800 comparisions in each alphabet 
'''
class OMNIGLOToneshot_trails(Dataset):
    def __init__(self, dataPath, transform=None):
        np.random.seed(10)
        self.alphabet_num=0
        super(OMNIGLOToneshot_trails, self).__init__()
        trials=self.generate_Oneshot_trails(dataPath,transform)
        
    def generate_Oneshot_trails(self,eval_path,way=20,transform=None):
        self.trails=dict()
        self.way=way
        for alph in range(0,10):
            self.trials[str(alph)]=[]
            self.trials[str(alph)]['train_img']=[]
            self.trials[str(alph)]['test_img']=[]       
            #20 way within trials of alphabets in the evaluation set
            #10-alphabets during evaluation and 10-alphabets during testing
            for repeat in range(0,2): #for each alphabet this process is repeated 2 times
                self.trials[str(alph)]['train_img'][repeat]=[]
                self.trials[str(alph)]['test_img'][repeat]=[]
                alpha_path=os.path.join(eval_path,os.listdir(eval_path)[str(alph)])
                characters=random.sample(range(0,len(os.listdir(alpha_path))),20)#draw 20 distint characters
                char_paths=[os.listdir(alpha_path)[i] for i in characters]
                drawers=random.sample(range(0,20),2)
                for i in range(20): #collect 20 characters of drawer-1 and pair with each character of drawer 2
                    image_folder_path=os.path.join(alpha_path,char_paths[i])
                    test_img=Image.open(os.listdir(image_folder_path)[drawers[0]]).convert('L')
                    train_img=Image.open(os.listdir(image_folder_path)[drawers[1]]).convert('L')
                    if transform!=None:
                        test_img=transform(test_img)
                        train_img=transform(train_img)
                    else:
                        self.trials[str(alph)]['train_img'][repeat].append(train_img)
                        self.trials[str(alph)]['test_img'][repeat].append(test_img)

        return self.trials
    def __len__():
        return 20 
    


    def __getitem__(self,index):
        '''index is ignored as we are returning a 2 lists(20 images each)  in one iteration
            For each iteration it will return 2 lists with 20 images each(of 10 different classes) [20 one shot trails]
            Each one-shot trails consist of comparing an image in one list with  20 images in other list ''' 
        alph=random.randint(0,9)
        num=random.randint(0,1)
        return self.trials[str(alph)]['test_img'][num],self.trials[str(alph)]['train_img'][num]


    def Omniglot_oneshot_acc(net,dataloader,cuda_check):
        right=0
        error=0
        output=dict()

        for test_img1_batch,test_img2_batch,test_label in dataloader:
            for test_label,test_img1 in enumerate(test_img1_batch,0):
                        for test_img2 in test_img2_batch:

                            if cuda_check:
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
    OMNIGLOToneshot_trails(Dataset)

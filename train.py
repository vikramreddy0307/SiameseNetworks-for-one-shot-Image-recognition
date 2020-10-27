import torch
from torch.autograd import Variable

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torchvision.datasets as datasets

import sys,argparse,pickle,os
import numpy as np 

from siamese import SiameseNetworks
from dataset_preparation import Omniglot_Train,Omniglot_Test,Omniglot_Validation
from Omniglot_oneshot import OMNIGLOToneshot_trails
from Mnist_oneshot import MINSToneshot_trails


parser = argparse.ArgumentParser()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")



parser.add_argument('--GPU',default=0,type=int,help='No GPU available')
parser.add_argument('--epochs',default=200,type=int,help='No GPU available')

parser.add_argument('--train_path',type=dir_path,default='data/omniglot-py/images_background')
parser.add_argument('--test_path',type=dir_path,default='data/omniglot-py/images_evaluation')
parser.add_argument('--way',default=20,type=int,help='way in MNIST one-shot task')

parser.add_argument('--one_shot_test',default=400,type=int,help='one shot learning trails to evaluate the accuracy')
parser.add_argument("--show_every", default=10, type=int,help= "show result after each show_every iter.")
parser.add_argument("--save_every", default=100,type=int, help="save model after each save_every iter.")
parser.add_argument("--test_every", default=100,type=int,help= "test model after each test_every iter.")
parser.add_argument("--workers", default=4,type=int, help="number of dataLoader workers")
parser.add_argument('--batch_size',default=128,type=int,help="Depends on the data size")
parser.add_argument('--learning_rate',default=0.0004,type=float,help="between 0.0001 and 0.1")
parser.add_argument('--iter',default=200,type=int,help="max number of iteration acc to paper")
parser.add_argument('--model_path',type=dir_path,default='model',help='path to store model')
parser.add_argument('--gpu_ids',default="0",help="Number of gpu's you want to use")

args = parser.parse_args()

affine_transform=transforms.Compose([transforms.RandomAffine(15),
    transforms.ToTensor()])


os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids
print('using gpu:',args.gpu_ids,'to train')
training_data=Omniglot_Train(args.train_path,transform=affine_transform)
validation_data=Omniglot_Validation(args.test_path,transform=transforms.ToTensor())






train_loader=DataLoader(training_data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
valid_loader=DataLoader(validation_data,batch_size=args.way,shuffle=False,num_workers=args.workers)



#Binary cross-entropy is for multi-label classifications, whereas 
#categorical cross entropy is for multi-class classification
loss=torch.nn.BCEWithLogitsLoss(size_average=True)

net=SiameseNetworks()
if  len(args.gpu_ids.split(','))>1:
    net=torch.nn.DataParallel(net)
if args.GPU:
	net.cuda()
net.train()
optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate)

optimizer.zero_grad()
train_loss=[]
val_acc=[]
OMNIGLOT_oneshot_acc=[]
MNIST_oneshot_acc=[]

val_loss=[]
for epoch in range(args.epochs):
    for  img1,img2,label in  train_loader :
    	if args.cuda:
    		img1,img2,label=Variable(img1.cuda()),Variable(img2.cuda()),Variable(label.cuda())
        else:
            img1,img2,label=Variable(img1),Variable(img2),Variable(label)
        optimizer.zero_grad()
        output=net.forward(img1,img2)
        val_loss+=loss.item()
        loss.backward()
        optimizer.step()
        if epoch % args.show_every == 0:
            print('Epoch-',epoch, 'val_loss-',val_loss/args.show_every)
            val_loss = 0
            if epoch%args.save_every==0:
                torch.save(net.state_dict(),args.model_path+'/model-'+str(epoch+1)+'.pt')
                if epoch%args.test_every==0:
                    right=error=0
                    for _, (test1, test2,label) in enumerate(valid_loader, 1):
                        if args.cuda: test1, test2 = test1.cuda(), test2.cuda()
                        test1, test2 = Variable(test1), Variable(test2)
                        predictions = net.forward(test1, test2).data.cpu().numpy()
                        for i in range(len(predictions)): 
                            if predictions[i] == label[i]:
                            	right += 1
                            else: error += 1
            print('*'*70)
            print('Epoch-',epoch,'correct-', right,'wrong-', error,'Acccuracy on Validation data-',right*1.0/(right+error))
            print('*'*70)
            val_acc.append(right*1.0/(right+error))


        

        
    train_loss.append(val_loss)
with open('train_loss', 'wb') as f:
    pickle.dump(train_loss, f)


'''
						Accuracy on test data(10 alphabets, 4 drawers)

'''

testing_data=Omniglot_Test(args.test_path,transform=transforms.ToTensor())
test_loader=DataLoader(testing_data,batch_size=args.way,shuffle=False,num_workers=args.workers)
right=error=0
for _, (test1, test2,label) in enumerate(test_loader, 1):
	if args.cuda: test1, test2 = test1.cuda(), test2.cuda()
    test1, test2 = Variable(test1), Variable(test2)
    predictions = net.forward(test1, test2).data.cpu().numpy()
    for i in range(len(predictions)): 
        if predictions[i] == label[i]:
        	right += 1
        else: error += 1
print('Accuracy on test data -',right/(right+error))




''' 
						One-shot accuracy calculation

'''

OMNIGLOTonsehot_test=OMNIGLOToneshot_trails(args.test_path,transform=transforms.ToTensor())
MNISToneshot_test=MINSToneshot_trails(transform=transforms.ToTensor())

''' In single iteration of one-shot OMNIGLOT  loader it will 
	return 2 list containing 20 images of 10 classes in each
	That is the reason why batch size=1    ''' 


OMNIGLOT_test_loader=DataLoader(test_set,batch_size=1,shuffle=False,num_workers=args.workers)

''' In single iteration of MNIST loader it will return2 lists containing 
    10 images each 10 one-shot trials(each one shot trial is
    comparision of one image with 10 other images) '''

MNIST_test_loader=DataLoader(MNIST_test_set,batch_size=1,shuffle=Flase,num_workers=args.workers)



OMNIGLOToneshot_acc=OMNIGLOToneshot_trails.Omniglot_oneshot_acc(net,OMNIGLOT_test_loader,args.cuda)
print(' one shot Test accuracy on OMNIGLOT ',OMNIGLOToneshot_acc)
MNISToneshot_acc=MNISToneshot_trails.Omniglot_oneshot_acc(net,MNIST_test_loader,args.cuda)
print(' one shot Test accuracy on MNIST ',MNISToneshot_acc)



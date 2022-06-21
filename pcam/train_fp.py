import argparse
import logging
import os
import random
import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
from custom_models import resnet20_fp

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

exp_num = 0
out_dir = os.path.join('out', str(exp_num))
while(os.path.exists(out_dir)):
    exp_num = exp_num + 1
    out_dir = os.path.join('out', str(exp_num))
print(out_dir, ' does NOT exists')
os.mkdir(out_dir)

parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (PCam)")
# data and model
parser.add_argument('--dataset', type=str, default='pcam_rgb224', choices=('pcam_rgb224','pcam_gs224', 'pcam_rgb224_sp200', 'pcam_gs224_sp200', 'mhist','mhist_gs', 'mhist_rgb_sp', 'mhist_gs_sp'), help='dataset to use variations of PCam')
parser.add_argument('--arch', type=str, default='resnet20_fp', help='model architecture')
parser.add_argument('--num_workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

# training settings
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--optimizer_m', type=str, default='SGD', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--lr_m', type=float, default=1e-1, help='learning rate for model parameters')
parser.add_argument('--lr_m_end', type=float, default=0.0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--decay_schedule_m', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for model parameters')
parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')

# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
#parser.add_argument('--log_dir', type=str, default='')
args = parser.parse_args()
arg_dict = vars(args)

log_dir = out_dir
### make log directory
if(not os.path.exists(os.path.join(log_dir, 'checkpoint'))):
    os.makedirs(os.path.join(log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')

### GPU setting
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

### train/test datasets
if(args.dataset == 'mhist'):
    mean = [0.7378, 0.6486, 0.7752]
    std = [0.1879, 0.2293, 0.1628]
elif(args.dataset == 'mhist_gs'):
    mean = [0.6898, 0.6898, 0.6898] 
    std = [0.2081, 0.2081, 0.2081]
elif(args.dataset == 'mhist_gs_sp'):
    mean = [0.3234, 0.3234, 0.3234] 
    std = [0.2762, 0.2762, 0.2762]
elif(args.dataset == 'mhist_rgb_sp'):
    mean = [0.3944, 0.3124, 0.4292] 
    std = [0.3092, 0.2591, 0.3272]
elif(args.dataset == 'pcam_rgb224'):
    mean = [0.7008, 0.5384, 0.6916]
    std = [0.1818, 0.2008, 0.1648]
elif(args.dataset == 'pcam_gs224'):
    mean = [0.6044, 0.6044, 0.6044]
    std = [0.1799, 0.1799, 0.1799]
elif(args.dataset == 'pcam_gs224_sp200'):
    mean = [0.3386, 0.3386, 0.3386]
    std = [0.2281, 0.2281, 0.2281]
elif(args.dataset == 'pcam_rgb224_sp200'):
    mean = [0.4674, 0.3160, 0.4622]
    std = [0.2650, 0.2166, 0.2382]
else:
    raise ValueError('ERROR: dataset name not found: ', args.dataset)

transform_train = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor()
             #,transforms.Normalize(mean, std)
            ])
transform_test = transforms.Compose([
            transforms.ToTensor()
            #, transforms.Normalize(mean, std)
            ])

data_dir = '/work/deogun/alali/data/'
data_dir = os.path.join(data_dir, args.dataset)
args.num_classes = 2
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                transform=transform_train)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), 
            transform=transform_test)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'),
                            transform=transform_test)

#analyze one sample for zero numbers
#sample_img = glob.glob(os.path.join(data_dir, 'train', '*', '*.png'))[0]
#print('sample img path: ', sample_img)
#images = []
#pil_images = []
#for s in sample_img:
#pil_im = Image.open(sample_img)
#pil_images.append(pil_im)
#im = np.array(pil_im)
#images.append(im)
#print('np im shape: {}, max={}, min={}, type={}'.format(im.shape, np.amax(im), np.amin(im), im.dtype))
#np_images = np.array(images)
#print('np images shape: {}, max={}, min={}, type={}'.format(np_images.shape, 
#    np.amax(np_images), np.amin(np_images), np_images.dtype))
#non_zero = np.count_nonzero(im)
#total_shape = np.prod(list(im.shape))
#zero_count = total_shape - non_zero
#print('total_shape - non_zero = zero count = {} - {} = {}'.format(total_shape, non_zero, zero_count))

#torch_img = transform_train(pil_im)
#print('torch_img shape={}, type={}'.format(torch_img.shape, torch_img.dtype))
#non_zero = torch.count_nonzero(torch_img)
#print('torch non zero count: ', non_zero)
#np_img = torch_img.numpy()
#np_img = np.swapaxes(np_img, 0, 2)*225
#np_img = np_img.astype('uint8')
#print('np img: {}, max={}, min={}, type={}'.format(np_img.shape, np.amax(np_img), np.amin(np_img), np_img.dtype))
#pil_img = Image.fromarray(np_img)
#print('pil img: ', pil_img)
#pil_img.save(args.dataset+'_sample_torch.png')

print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
print('train datasets: {} \nval: {} \ntest: {}'.format(train_dataset, val_dataset, test_dataset))
# Create training and validation dataloaders

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        batch_size = args.batch_size//2,
        shuffle=False,
        )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=args.num_workers)
#print('train loader: ', list(train_loader))
#print('test loader: ', list(test_loader))
### initialize model
model_class = globals().get(args.arch)
#print('model class: ', model_class)
model = model_class(args)
model.to(device)
#print('model arch: \n', model)
#i = 1
#for name, layer in model.named_modules():
#    if(isinstance(layer, nn.Conv2d)):
#        print('{}- name: {}\nweights shape={}\nweights min={0:0.3f}\nweights mean={0:0.3f}'.format(i, name, layer.weight.shape, torch.min(layer.weight).item(), torch.mean(layer.weight).item()))
#    elif(isinstance(layer, nn.Linear)):
#        print(i, 'name: ', name, ' layer: ', layer)
#    i = i + 1

#summary(model, (3, 224, 224))

num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))
#count zero parameters
#zeros = 0
#TODO: try loading state dict, check num of zeros
#for i, p in enumerate(model.parameters()):
#    print('{}- param shape: {}'.format(i, p.shape))
#    print('mean={0:0.3f}, min={0:0.3f}'.format(torch.mean(p).item(),torch.min(p).item()))
#    if p is not None:
#        zeros += torch.sum((p == 0).int()).item()
        #print('Number of zeros: ', zeros)
#print('Number of zeros in the model parameters: ', zeros)


### initialize optimizer, scheduler, loss function
# optimizer for model params
if args.optimizer_m == 'SGD':
    optimizer_m = torch.optim.SGD(model.parameters(), lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer_m == 'Adam':
    optimizer_m = torch.optim.Adam(model.parameters(), lr=args.lr_m, weight_decay=args.weight_decay)
    
# scheduler for model params
if args.lr_scheduler_m == "step":
    if args.decay_schedule_m is not None:
        milestones_m = list(map(lambda x: int(x), args.decay_schedule_m.split('-')))
    else:
        milestones_m = [args.epochs+1]
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, gamma=args.gamma)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir)

### train
total_iter = 0
best_acc = 0
iter_print = 0
train_loss_list = []
val_loss_list = []
if('pcam' in args.dataset):
    iter_print = 500
else:
    iter_print = 10
for ep in range(args.epochs):
    print('========= Epoch: [{}/{}] ========='.format(ep, args.epochs))
    model.train()
    writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)
    print('train loop for {} iterations'.format(len(train_loader)))
    train_epoch_loss_list = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        if(i == 0):
            #np_images = images.cpu().detach().numpy()
            #image = images[0] We want number of zeros in the entire batch
            #print('images shape: ', images.shape)
            #print('np images shape: {}, type={}, min={}, max={}'.format(np_images.shape, np_images.dtype, np.amin(np_images), np.amax(np_images)))
            #non_zeros = np.count_nonzero(np_images)
            #total_shape = np.prod(np_images.shape)
            #num_zeros = total_shape - non_zeros
            #print('count zero np => total - nonzero = {} - {} = {}'.format(total_shape, non_zeros, num_zeros))
            t_non_zeros = torch.count_nonzero(images)
            t_total_shape = torch.prod(torch.tensor(images.shape))
            t_num_zeros = t_total_shape - t_non_zeros
            print('torch images shape: {}, type={}, min={}, max={}'.format(images.shape, images.dtype, torch.min(images), torch.max(images)))
            print('count zeros torch: total - nonzero = {} - {} = {}'.format(t_total_shape, 
                t_non_zeros, t_num_zeros))
        optimizer_m.zero_grad()
        pred = model(images)
        loss_t = criterion(pred, labels)
        if(i % iter_print == 0):
            print(i, '- batch shape: ', images.shape, ' loss: ', loss_t.item()) 
        train_epoch_loss_list.append(loss_t.item())
        loss = loss_t
        loss.backward()
        
        optimizer_m.step()
        writer.add_scalar('train/loss', loss.item(), total_iter)
        total_iter += 1
    train_loss_list.append(np.mean(train_epoch_loss_list))
    print('epoch train loss: ', np.mean(train_epoch_loss_list))
    scheduler_m.step()

    with torch.no_grad():
        #model.eval()
        #correct_classified = 0
        #total = 0
        #TODO: why another train_loader?? ==> validation
        #for i, (images, labels) in enumerate(val_loader):
        #    images = images.to(device)
        #    labels = labels.to(device)
            #print('validation i={}, images shape={}, labels shape={}'.format(i, images.shape, labels.shape))
        #    pred = model(images)
        #    _, predicted = torch.max(pred.data, 1)
        #    total += pred.size(0)
        #    correct_classified += (predicted == labels).sum().item()
        #    if(i % 200 == 0):
        #        print('val i = {}, correct_classified={}'.format(i, correct_classified))
        #val_acc = correct_classified/total*100
        #print('val accuracy: ', val_acc)
        #writer.add_scalar('val/acc', val_acc, ep)

        model.eval()
        correct_classified = 0
        total = 0
        print('loop validation for {} iterations'.format(len(val_loader)))
        val_epoch_loss_list = []
        for j, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            val_loss = criterion(pred, labels)
            val_epoch_loss_list.append(val_loss.item())
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
            if(j % iter_print == 0):
                print('val j = {}, correct classified = {}'.format(j, correct_classified)) 
        val_acc = correct_classified/total*100
        print("Current epoch: {:03d}".format(ep), "\t Val accuracy:", val_acc, "%")
        print('val loss: ', np.mean(val_epoch_loss_list))
        logging.info("Current epoch: {:03d}\t Val accuracy: {}%".format(ep, val_acc))
        writer.add_scalar('val/acc', val_acc, ep)
        val_loss_list.append(np.mean(val_epoch_loss_list))
        torch.save({
            'epoch':ep,
            'model':model.state_dict(),
            'optimizer_m':optimizer_m.state_dict(),
            'scheduler_m':scheduler_m.state_dict(),
            'criterion':criterion.state_dict()
        }, os.path.join(log_dir,'checkpoint/last_checkpoint.pth'))
        if(val_acc > best_acc):
            best_acc = val_acc
            torch.save({
                'epoch':ep,
                'model':model.state_dict(),
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'criterion':criterion.state_dict()
            }, os.path.join(log_dir,'checkpoint/best_checkpoint.pth'))  
    

#PLOT train val loss curve
plt.figure()
plt.plot(train_loss_list, 'g--', label='training loss')
plt.plot(val_loss_list, '-', label='validation loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
fig_path = '{}/exp_{}_train_valid_loss.png'.format(out_dir, args.dataset)
plt.savefig(fig_path, dpi=300)
plt.close()
print('train valid loss curve figure saved in path: ', fig_path)

##### Testing/Inference Phase ####
### Test accuracy @ last checkpoint
trained_model = torch.load(os.path.join(log_dir,'checkpoint/last_checkpoint.pth'))
model.load_state_dict(trained_model['model'])

num_zeros = 0
#TODO: check values of state dict
for i, (name, param) in enumerate(model.named_parameters()):
    if('conv' in name or 'linear' in name):
        print('{}- name: {}, shape={}'.format(i, name, param.shape))
        t_non_zeros = torch.count_nonzero(param)
        t_total_shape = torch.prod(torch.tensor(param.shape))
        total_el = torch.numel(param)
        t_num_zeros = t_total_shape - t_non_zeros
        print('min={}, max={}, number elements={}'.format(torch.min(images), torch.max(images), total_el))
        print('count zeros: total - nonzero = {} - {} = {}'.format(t_total_shape,
                      t_non_zeros, t_num_zeros))
#        print(name, '======================================================')
#        for p in param:
#            print('p shape: ', p.shape, ' mean: ', torch.mean(p).item(), ' min: ', torch.min(p).item())
#            for item in torch.flatten(p):
#                if(torch.isclose(item, torch.tensor(0.0))):
#                    print("close to zero")
#                    num_zeros += 1
#            nonz = torch.count_nonzero(p).item()
#            numz = p.numel() - nonz
#            print('num of zeros: ', numz, ' nonzero= ', nonz, ' total: ', p.numel())
#            if(numz > 1):
#                raise ValueError("Finally: found zero!!")

print("The last checkpoint is loaded")
logging.info("The last checkpoint is loaded")
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    correct_classified = 0
    total = 0
    print('loop testing for {} iterations'.format(len(test_loader)))
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        y_true.append(labels.item())
        pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        y_pred.append(predicted.item())
        total += pred.size(0)
        correct_classified += (predicted == labels).sum().item()
    test_acc = correct_classified/total*100
    print("Test accuracy: {}%".format(test_acc))
    logging.info("Test accuracy: {}%".format(test_acc))

from sklearn.metrics import classification_report as class_rep
print(class_rep(y_pred=y_pred, y_true=y_true))
print('out dir', out_dir)

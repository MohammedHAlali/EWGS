import argparse
import logging
import os
import random
import glob
from PIL import Image
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import custom_models
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


print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
print('train datasets: {} \nval: {} \ntest: {}'.format(train_dataset, val_dataset, test_dataset))
# Create training and validation dataloaders

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        batch_size = args.batch_size,
        shuffle=False,
        )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=args.num_workers)

# Test time for loading training data ##
#source: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
#start_time = time()
#for i, data in enumerate(train_loader):
#    pass
#end_time = time()
#print('Simulated training \nfinished with {} seconds, num_workers={}'.format(end_time-start_time, 
#    args.num_workers))
#exit()
#print('train loader: ', list(train_loader))
#print('test loader: ', list(test_loader))
### initialize model
model_class = globals().get(args.arch)
#print('model class: ', model_class)
model = model_class(args)
model.to(device)
print('model arch: \n', model)
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
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, 
            gamma=args.gamma, verbose=True)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, 
            eta_min=args.lr_m_end, verbose=True)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir)

### train
total_iter = 0
best_acc = 0
iter_print = 0
best_epoch = 0
train_loss_list = []
val_loss_list = []
if('pcam' in args.dataset):
    iter_print = 1000
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
        optimizer_m.zero_grad()
        output = model(images)
        pred = output[0]
        if(ep == args.epochs-1):
            print('================= Num of Zero Calculations ===================')
            t_non_zeros = torch.count_nonzero(images)
            t_total_shape = torch.prod(torch.tensor(images.shape))
            t_num_zeros = t_total_shape - t_non_zeros
            print('input images shape: {}, type={}, min={}, max={}'.format(images.shape, images.dtype, torch.min(images), torch.max(images)))
            print('input count zeros torch: total - nonzero = {} - {} = {}'.format(t_total_shape, 
                t_non_zeros, t_num_zeros))
            #Analyze first layer in the model
            if(type(output[1]) == nn.Conv2d):
                w = output[1].weight
                print('first conv2d: {}, weight shape={}'.format(output[1], w.shape))
                layer_out = output[1](images)
                w_non_zero = torch.count_nonzero(w)
                w_total_shape = torch.prod(torch.tensor(w.shape))
                w_num_zero = w_total_shape - w_non_zero
                w_mean = torch.mean(w).item()
                w_min = torch.min(w).item()
                print('weights num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                    w_total_shape, w_non_zero, w_num_zero, w_mean, w_min))
                l_out = layer_out
                #print('layer out shape: ', l_out.shape)
                l_non_zero = torch.count_nonzero(l_out)
                l_total_shape = torch.prod(torch.tensor(l_out.shape))
                l_num_zero = l_total_shape - l_non_zero
                l_mean = torch.mean(l_out).item()
                l_min = torch.min(l_out).item()
                print('layer out num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                    l_total_shape, l_non_zero, l_num_zero, l_mean, l_min))
            print('===== processing middle layers ======')
            seq_layer_input = None
            for m, layer in enumerate(output[2]):
                #if(m == 0):
                seq_layer_input = layer_out
                #print('seq#{} input shape: {}'.format(m, seq_layer_input.shape))
                layer_out = layer(seq_layer_input)
                #else:
                #    print('seq#{} input shape: {}'.format(m, layer_out.shape))
                #    layer_out = layer(layer_out)
                #print('seq#{} type: {}'.format(m,type(layer)))
                #print('layer = ', layer)
                #print('seq#{} output shape: {}'.format(m, layer_out.shape))
                block_layer_out = None
                for ii, block_layer in enumerate(layer):
                    print('== basic block#{} type: {}'.format(ii, type(block_layer)))
                    if(ii == 0):
                        #bring block input, feed it to first layer in the block
                        block_layer_in = seq_layer_input
                        print('== basic block#{} input shape: {}'.format(ii, block_layer_in.shape))
                        layer_out = block_layer(block_layer_in)
                        print('== basic block#{} output shape: {}'.format(ii, layer_out.shape))
                        block_layer_out = layer_out
                    elif(ii == len(layer)-1):
                        print('== last basic block#', ii)
                        block_layer_in = layer_out
                        print('== basic block#{} input shape: {}'.format(ii, layer_out.shape))
                        block_layer_out = block_layer(layer_out)
                        print('== basic block#{} output shape: {}'.format(ii, block_layer_out.shape))
                    else:
                        #print('== TODO: take layer_in from block_layer_out, DONE')
                        layer_out = block_layer_out
                        block_layer_in = layer_out
                        print('== basic block#{} input shape: {}'.format(ii, layer_out.shape))
                        layer_out = block_layer(layer_out)
                        print('== basic block#{} output shape: {}'.format(ii, layer_out.shape))
                    #print('==: ', inner_layer)
                    for iii, innest_layer in enumerate(block_layer.children()):
                        if(iii == 0):
                            # take input from the block input layer => feeding from outside
                            innest_layer_in = block_layer_in
                        else:
                            # take input from previous layer in the block => feeding from inside
                            innest_layer_in = layer_out

                        #print('==== innest layer#{} input shape: {}'.format(iii, innest_layer_in.shape))
                        if(type(innest_layer) == nn.Conv2d):
                            print('==== innest layer#{}, = {}'.format(iii, innest_layer))
                            #print('==== find zeros in ifmap')
                            #print('==== ifmap shape: ', innest_layer_in.shape)
                            i_non_zero = torch.count_nonzero(innest_layer_in)
                            i_total_shape = torch.prod(torch.tensor(innest_layer_in.shape))
                            i_num_zero = i_total_shape - i_non_zero
                            i_mean = torch.mean(innest_layer_in).item()
                            i_min = torch.min(innest_layer_in).item()
                            print('==== ifmap num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                                i_total_shape, i_non_zero, i_num_zero, i_mean, i_min))
                            #print('==== find zeros in filter fmap')
                            #print('==== weights shape: ', w.shape)
                            w = innest_layer.weight
                            w_non_zero = torch.count_nonzero(w)
                            w_total_shape = torch.prod(torch.tensor(w.shape))
                            w_num_zero = w_total_shape - w_non_zero
                            w_mean = torch.mean(w).item()
                            w_min = torch.min(w).item()
                            print('==== filter num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                                w_total_shape, w_non_zero, w_num_zero, w_mean, w_min))
                            layer_out = innest_layer(innest_layer_in)
                            #print('==== innest layer#{} output shape: {}'.format(iii, layer_out.shape))
                            o_non_zero = torch.count_nonzero(layer_out)
                            o_total_shape = torch.prod(torch.tensor(layer_out.shape))
                            o_num_zero = o_total_shape - o_non_zero
                            o_mean = torch.mean(layer_out).item()
                            o_min = torch.min(layer_out).item()
                            print('==== ofmap num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                                o_total_shape, o_non_zero, o_num_zero, o_mean, o_min))
                        else:
                            layer_out = innest_layer(innest_layer_in)
            # Analyze last two layers in the model            
            if(len(output[3]) == 2):
                layers = output[3]
                out = block_layer_out
                # get layer before Linear [DONE]
                print('n-1 layer: ', layers[0])
                #print('input shape: ', out.shape)
                out = F.avg_pool2d(out, out.size()[3])
                #print('avg pool output shape: ', out.shape)
                out = out.view(out.size(0), -1)
                #print('2D input shape: ', out.shape)
                out = layers[0](out)
                #print('out shape: ', out.shape)
                print('working on last linear layer, =', layers[1])
                print('input shape: ', out.shape)
                i_non_zero = torch.count_nonzero(out)
                i_total_shape = torch.prod(torch.tensor(out.shape))
                i_num_zero = i_total_shape - i_non_zero
                i_mean = torch.mean(out).item()
                i_min = torch.min(out).item()
                print('==== ifmap num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                    i_total_shape, i_non_zero, i_num_zero, i_mean, i_min))
                w = layers[1].weight
                print('layer weight shape: ', w.shape)
                w_non_zero = torch.count_nonzero(w)
                w_total_shape = torch.prod(torch.tensor(w.shape))
                w_num_zero = w_total_shape - w_non_zero
                w_mean = torch.mean(w).item()
                w_min = torch.min(w).item()
                print('==== filters num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                    w_total_shape, w_non_zero, w_num_zero, w_mean, w_min))
                print('output shape: ', out.shape)
                out = layers[1](out)
                print('linear out shape: ', out.shape)
                o_non_zero = torch.count_nonzero(out)
                o_total_shape = torch.prod(torch.tensor(out.shape))
                o_num_zero = o_total_shape - o_non_zero
                o_mean = torch.mean(out).item()
                o_min = torch.min(out).item()
                print('==== ofmap num zero: total - nonzero = {}-{}={}, mean={:.2f}, min={}'.format(
                    o_total_shape, o_non_zero, o_num_zero, o_mean, o_min))
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
        model.eval()
        correct_classified = 0
        total = 0
        print('loop validation for {} iterations'.format(len(val_loader)))
        val_epoch_loss_list = []
        for j, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            pred = output[0]
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
            best_epoch = ep
            torch.save({
                'epoch':ep,
                'model':model.state_dict(),
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'criterion':criterion.state_dict()
            }, os.path.join(log_dir,'checkpoint/best_checkpoint.pth'))  
    
print('best epoch: ', best_epoch)
#PLOT train val loss curve
plt.figure()
plt.plot(train_loss_list, 'g--', label='training loss')
plt.plot(val_loss_list, '-', label='validation loss')
plt.title('Training and Validation Loss, {}'.format(args.dataset))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
fig_path = '{}/exp_{}_train_valid_loss.png'.format(out_dir, args.dataset)
plt.savefig(fig_path, dpi=300)
plt.close()
print('train valid loss curve figure saved in path: ', fig_path)

##### Testing/Inference Phase ####
### Test accuracy @ last checkpoint
trained_model = torch.load(os.path.join(log_dir,'checkpoint/best_checkpoint.pth'))
model.load_state_dict(trained_model['model'])

#source: https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/2
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.linear.register_forward_hook(get_activation('linear'))
#output = model(x)
print('activation[linear]: ', activation[linear])
"""
a_batch_img, _ = next(iter(train_loader))
print('a batch img shape: ', a_batch_img.shape)
#a_batch_img = transforms.ToTensor()(a_batch_img)
print('type={}, min={}, max={}'.format(torch.min(a_batch_img), torch.max(a_batch_img), type(a_batch_img)))
non_zeros = torch.count_nonzero(a_batch_img)
total_shape = torch.prod(torch.tensor(a_batch_img.shape)) #or use numel()
num_zeros = total_shape - non_zeros
print('Input fmap: count zeros: total - nonzero = {} - {} = {}'.format(total_shape, non_zeros, num_zeros))
images_out = a_batch_img.to(device)

for i, layer in enumerate(model.children()):
    #feed input to layer
    #take output, set as input for next layer in the loop
    if(type(layer) == nn.Conv2d):
        print(i, '- layer: ', layer)
        print('input shape: ', images_out.shape)
        images_out = layer(images_out)
        print('layer weights shape: ', layer.weight.shape)
        print('layer out shape: ', images_out.shape)
        non_zeros = torch.count_nonzero(images_out)
        w_non_zeros = torch.count_nonzero(layer.weight)
        total_shape = torch.prod(torch.tensor(images_out.shape)) #or use numel()
        w_total_shape = torch.prod(torch.tensor(layer.weight.shape))
        num_zeros = total_shape - non_zeros
        w_num_zeros = w_total_shape - w_non_zeros
        print('Weights: min={}, max={}'.format(torch.min(layer.weight), torch.max(layer.weight)))
        print('Weights: count zeros: w_total - w_nonzero = {} - {} = {}'.format(
            w_total_shape, w_non_zeros, w_num_zeros))
        print('Output: min={}, max={}'.format(torch.min(images_out), torch.max(images_out)))
        print('Output: count zeros: total - nonzero = {} - {} = {}'.format(total_shape, non_zeros, num_zeros))
        #feed to next layer
    elif(type(layer) == nn.Sequential):
        for j, a_layer in enumerate(layer):
            if(type(a_layer) == custom_models.BasicBlock):
                #print('BasicBlock layer: ', a_layer)
                for k, aa_layer in enumerate(a_layer.children()):
                    print('input shape: ', images_out.shape)
                    images_out = aa_layer(images_out)
                    print('min={}, max={} layer type={}'.format(torch.min(images_out), 
                        torch.max(images_out), type(aa_layer)))
                    if(type(aa_layer) == nn.Conv2d):
                        print('{}.{}.{}- layer: {}'.format(i,j,k, aa_layer))
                        print('layer weights shape: ', aa_layer.weight.shape)
                        w_non_zeros = torch.count_nonzero(aa_layer.weight)
                        w_total_shape = torch.prod(torch.tensor(aa_layer.weight.shape))
                        w_num_zeros = w_total_shape - w_non_zeros
                        print('Weights: count zeros. w_total - w_nonzero = {} - {} = {}'.format(
                                                        w_total_shape, w_non_zeros, w_num_zeros))
                        non_zeros = torch.count_nonzero(images_out)
                        total_shape = torch.prod(torch.tensor(images_out.shape))
                        num_zeros = total_shape - non_zeros
                        print('layer out shape: ', images_out.shape)
                        print('Output: count zeros. total - nonzero = {} - {} = {}'.format(
                            total_shape, non_zeros, num_zeros))

                    else:
                        print('{}.{}.{}- not conv2d layer type: {}'.format(i,j,k, type(aa_layer)))
            else:
                print('{}.{}- not BasicBlock layer type: {}'.format(i,j, type(a_layer)))
    else:
        print('layer not conv2d and not sequential, type=', type(layer))
        print('input shape: ', images_out.shape)
        images_out = layer(images_out)
        print('min={}, max={}'.format(torch.min(images_out), 
            torch.max(images_out)))
"""
exit()
num_zeros = 0
#TODO: check values of state dict
#source:
#https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/2
#https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/27
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

#from: https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
# we will save the conv layer weights in this list
model_weights =[]
#we will save the conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
print('Number of model children: ', len(model_children))
#TODO: check named_parameters...
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
for c in conv_layers:
    print(c)
print('num of conv layers: ', len(conv_layers))
print('num of model weights: ', len(model_weights))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
a_batch_img, _ = next(iter(train_loader))
print('a batch img shape: ', a_batch_img.shape)
#a_batch_img = transforms.ToTensor()(a_batch_img)
print('type={}, min={}, max={}'.format(torch.min(a_batch_img), torch.max(a_batch_img), type(a_batch_img)))
non_zeros = torch.count_nonzero(a_batch_img)
total_shape = torch.prod(torch.tensor(a_batch_img.shape)) #or use numel()
num_zeros = total_shape - non_zeros
print('count zeros. total - nonzero = {} - {} = {}'.format(total_shape, non_zeros, num_zeros))
a_batch_img = a_batch_img.to(device)
#an_img = a_batch_img[0]
#an_img = an_img.to(device)
#print('an image shape: ', an_img.shape)
# Generate feature maps
outputs = []
names = []
for i, layer in enumerate(conv_layers):
    print(i, '- layer: ', str(layer))
    images_out = layer(a_batch_img)
    print('layer weights shape: ', layer.weight.shape)
    print('layer out shape: ', images_out.shape)
    non_zeros = torch.count_nonzero(images_out)
    total_shape = torch.prod(torch.tensor(images_out.shape)) #or use numel()
    num_zeros = total_shape - non_zeros
    print('min={}, max={}'.format(torch.min(images_out), torch.max(images_out)))
    print('count zeros. total - nonzero = {} - {} = {}'.format(total_shape, non_zeros, num_zeros))
    outputs.append(images_out)
    names.append(str(layer))
print('outputs list: ', outputs)
#print feature_maps
for feature_map in outputs:
    print('feature_map.shape: ', feature_map.shape)


exit()
print("The best checkpoint is loaded")
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
print('out dir', out_dir, ' dataset: ', args.dataset)

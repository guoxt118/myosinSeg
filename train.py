from networks.unet_att import UNet_att
from networks.unet import UNet
import argparse
import logging
import os
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from dataset_myosin import myosin_dataset, RandomGenerator



def trainer_myosin(args, model, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    db_train = myosin_dataset(base_dir=args.root_path,seq=args.data_seq, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.00001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / 4000) ** 0.9
            # lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        if (epoch_num+1) % 1 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
    writer.close()
    return "Training Finished!"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/myosin', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='myosin', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--att', default=True)
parser.add_argument('--max_epochs', type=int,default=120, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=288, help='input patch size of network input')
parser.add_argument('--opt', default='adam', choices=['sgd','adam'])
parser.add_argument('--data_seq',type = int, default=3, choices=[1,2,3,4,5])
args = parser.parse_args()

if __name__ == "__main__":
    save_path = "model/myosin"+"_"+str(args.att)+"_"+str(args.max_epochs)+"_"+str(args.batch_size)+"_"+str(args.base_lr) + "_" + str(args.img_size) + "_" + str(args.opt)+"_"+str(args.data_seq)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.att:
        net = UNet_att(n_classes=args.num_classes).cuda()
    else:
        net = UNet(n_classes=args.num_classes).cuda()
    trainer_myosin(args, net, save_path)

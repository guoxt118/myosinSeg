import argparse
import logging
import os
from networks.unet import UNet
from networks.unet_att import UNet_att
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_myosin import myosin_dataset
from utils import test_single

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/myosin', help='root dir for validation data')
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


def inference(args, model, test_save_path=None):
    db_test = myosin_dataset(base_dir=args.root_path, seq = args.data_seq, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single(image, label, model, classes=args.num_classes, test_save_path=test_save_path, case=case_name)
        metric_list += np.array(metric_i)
        # print('idx %d case %s mean_iou %f mean_acc %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        print('Mean class %d mean_iou %f mean_acc %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    iou = np.mean(metric_list, axis=0)[0]
    acc = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_iou : %f mean_acc : %f' % (iou, acc))
    return "Testing Finished!"


if __name__ == "__main__":
    save_path = "model/myosin"+"_"+str(args.att)+"_"+str(args.max_epochs)+"_"+str(args.batch_size)+"_"+str(args.base_lr) + "_" + str(args.img_size) + "_" + str(args.opt)+"_"+str(args.data_seq)
    print(save_path)
    if args.att:
        net = UNet_att(n_classes=args.num_classes).cuda()
    else:
        net = UNet(n_classes=args.num_classes).cuda()
    test_save_path = 'pred/predictions'+"_"+str(args.att)+"_"+str(args.max_epochs)+"_"+str(args.batch_size)+"_"+str(args.base_lr) + "_" + str(args.img_size) + "_" + str(args.opt)+"_"+str(args.data_seq)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)
    for i in range(0,args.max_epochs,1):
        net.load_state_dict(torch.load(os.path.join(save_path,"epoch_"+str(i)+".pth")))
        print(i)
        inference(args, net, test_save_path)



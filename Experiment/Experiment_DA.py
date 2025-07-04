import sys

sys.path.append("..")
import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yaml
from CMAPSS_Related.load_data_CMAPSS import da_get_cmapss_data_
from CMAPSS_Related.CMAPSS_Dataset import CMAPSSData
from N_CMAPSS_Related.N_CMAPSS_load_data import N_CMAPSSData, da_get_n_cmapss_data_
from XJTU_Related.load_data_XJTU import da_get_xjtu_data_, XJTUData
import shutil
from torch.utils.data import DataLoader

from Model.Transfomer_domain_adaptive import Transformer_domain,Discriminator,backboneDiscriminator

from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from Experiment.HTS_Loss_Function import HTSLoss
from Experiment.HTS_Loss_Function import Weighted_MSE_Loss, MSE_Smoothness_Loss
from tqdm import tqdm
from tool.Write_csv import *
import datetime

"""
This file only used for CMAPSS Datase
"""

def advLoss(source, target, device):

    sourceLabel = torch.ones(len(source)).double()
    targetLabel = torch.zeros(len(target)).double()
    Loss = nn.BCELoss()
    if device == 'cuda':
        Loss = Loss.cuda()
        sourceLabel, targetLabel = sourceLabel.cuda(), targetLabel.cuda()
    #print("sd={}\ntd={}".format(source, target))
    loss = Loss(source, sourceLabel) + Loss(target, targetLabel)
    return loss*0.5




class Exp_DA(object):
    def __init__(self, args):
        print('DA training')
        self.args = args

        self.device = self._acquire_device()

        # load CMAPSS dataset
        self.s_train_data, self.t_train_data, self.s_train_loader, self.t_train_loader, \
        self.s_vali_data, self.t_vali_data, self.s_vali_loader, self.t_vali_loader, \
        self.test_data, self.test_loader, self.input_feature = self._get_data()

        self._get_path()

        # build the Inception-Attention Model:
        self.model = self._get_model()
        self.D1 = Discriminator(self.args.input_length).double().to(self.device)
        self.D2 = backboneDiscriminator(self.args.input_length,self.args.d_model).double().to(self.device)


        # What optimisers and loss functions can be used by the model
        self.optimizer_dict = {"Adam": optim.Adam}
        self.criterion_dict = {"MSE": nn.MSELoss, "CrossEntropy": nn.CrossEntropyLoss, "WeightMSE": Weighted_MSE_Loss,
                               "smooth_mse": MSE_Smoothness_Loss}

    # choose device
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU: cuda:'+str(os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device


    def _get_path(self):
        if not os.path.exists('./logs/'):
            os.makedirs('./logs/')

        exp_id = self.args.save_path
        # save address
        self.path = './logs/' + self.args.Data_id_CMAPSS
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.model_path = self.path + '/' + self.args.model_name
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if exp_id is not None and exp_id != 'None' and exp_id != 'none':
            self.save_path = self.model_path + '/' + exp_id
            if self.args.train:
                if os.path.exists(self.save_path):
                    shutil.rmtree(self.save_path)
                os.makedirs(self.save_path)


        else:
            # 如为None则自动加一
            path_list = os.listdir(self.model_path)
            if path_list == []:
                self.save_path = self.model_path + '/exp0'

            else:
                path_list = [int(idx[3:]) for idx in path_list]
                self.save_path = self.model_path + '/exp' + str(max(path_list) + 1)

            os.makedirs(self.save_path)

    def _load_checkpoint(self):
        self.checkpoint_dir = self.model_path + '/' + self.args.resume_path + '/best_checkpoint.pth'
        if os.path.exists(self.checkpoint_dir):
            check_point = torch.load(self.checkpoint_dir)
            self.model.load_state_dict(check_point)
        else:
            raise 'checkpoint is not exists'

        yaml_dir = self.model_path + '/' + self.args.resume_path + '/hparam.yaml'
        with open(yaml_dir, 'r', encoding='utf-8') as f:
            yml = yaml.load(f.read(), Loader=yaml.FullLoader)
            key2load = [k for k in yml.keys()]
            for k in key2load:
                # 重赋值
                exec('self.args.' + k + '=yml[k]')

    # ------------------- function to build model -------------------------------------
    def _get_model(self):

        if self.args.model_name == 'Transformer_domain':
            model = Transformer_domain(self.args, input_feature=self.input_feature)


        print("Parameter :", np.sum([para.numel() for para in model.parameters()]))

        return model.double().to(self.device)

    # --------------------------- select optimizer ------------------------------
    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError

        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # ---------------------------- select criterion --------------------------------
    def _select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError

        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    # ------------------------ get Dataloader -------------------------------------
    # 要同时输入source 和 Target 意味着 输出也是要两个
    #  funnction of load CMPASS Dataset
    def _get_data(self, flag="train"):
        args = self.args

        # train and validation dataset
        if self.args.dataset_name == 'CMAPSS':
            s_x_train, t_x_train, s_x_vali, t_x_vali, t_y_train, t_y_vali, X_test, y_test = da_get_cmapss_data_(data_path=args.data_path_CMAPSS,
                                                                            s_id=args.source_domain,
                                                                         t_id=args.target_domain,
                                                                         sequence_length=args.input_length,
                                                                         MAXLIFE=args.MAXLIFE_CMAPSS,
                                                                         is_difference=args.is_diff,
                                                                         normalization=args.normalization_CMAPSS,
                                                                         validation=args.validation)
            self.max_life = args.MAXLIFE_CMAPSS

        if self.args.dataset_name == 'N_CMAPSS':
            s_x_train, t_x_train, s_x_vali, t_x_vali, t_y_train, t_y_vali, X_test, y_test, self.max_life = da_get_n_cmapss_data_(args, source=args.source_domain, target=args.target_domain)

        if self.args.dataset_name == 'XJTU':
            s_x_train, t_x_train, s_x_vali, t_x_vali, t_y_train, t_y_vali, X_test, y_test = da_get_xjtu_data_(pre_process_type="Vibration",
                root_dir='./XJTU/XJTU-SY_Bearing_Datasets/35Hz12kN',
                train_bearing_data_set=["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
                test_bearing_data_set=["Bearing1_1"],
                STFT_window_len=256,
                STFT_overlap_num=32,
                window_length=args.input_length,
                validation_rate=0.1)
            self.max_life = 1

        s_train_data_set = eval(self.args.dataset_name+'Data')(s_x_train, t_y_train)
        t_train_data_set = eval(self.args.dataset_name+'Data')(t_x_train, t_y_train)

        s_vali_data_set = eval(self.args.dataset_name+'Data')(s_x_vali, t_y_vali)
        t_vali_data_set = eval(self.args.dataset_name+'Data')(t_x_vali, t_y_vali)

        test_data_set = eval(self.args.dataset_name+'Data')(X_test, y_test)

        input_fea = X_test.shape[-1]


        s_train_data_loader = DataLoader(dataset=s_train_data_set,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=True)

        t_train_data_loader = DataLoader(dataset=t_train_data_set,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=True)

        s_vali_data_loader = DataLoader(dataset=s_vali_data_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=True)

        t_vali_data_loader = DataLoader(dataset=t_vali_data_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=True)

        test_data_loader = DataLoader(dataset=test_data_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=False)

        return s_train_data_set, t_train_data_set, s_train_data_loader, t_train_data_loader, \
               s_vali_data_set, t_vali_data_set, s_vali_data_loader, t_vali_data_loader, test_data_set, test_data_loader, input_fea


    def save_hparam(self):
        # args: args from argparse return
        value2save = {k: v for k, v in vars(self.args).items() if not k.startswith('__') and not k.endswith('__')}
        # 删除影响使用的参数，以便加载
        del_key = ['train', 'resume', 'save_path', 'resume_path', 'batch_size', 'train_epochs', 'learning_rate']
        for key in del_key:
            del value2save[key]

        with open(os.path.join(self.save_path, 'hparam.yaml'), 'a+') as f:
            f.write(yaml.dump(value2save))


    def train(self, save_path):

        # save address
        path = './logs/' + save_path
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = path + '/' + self.args.model_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # how many step of train and validation:
        train_steps = len(self.t_train_loader)
        vali_steps = len(self.t_vali_loader)
        print("train_steps: ", train_steps)
        print("validaion_steps: ", vali_steps)

        # initial early stopping
        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)

        # initial learning rate
        learning_rate_adapter = adjust_learning_rate_class(self.args, True)
        # choose optimizer
        model_optim = self._select_optimizer()

        # choose loss function
        loss_criterion = torch.nn.MSELoss()

        if self.args.resume:
            print('load checkpoint')
            self._load_checkpoint()
        else:
            print('random init')

        if self.args.train:
            # training process
            print("start training")
            # TODO 保存模型的超参数
            self.save_hparam()
            for epoch in range(self.args.train_epochs):
                start_time = time()

                iter_count = 0
                train_loss = []

                self.model.train()

                for _, (unpacked1, unpacked2) in enumerate(tqdm(zip(self.s_train_loader, self.t_train_loader), total=min(len(iter(self.s_train_loader)), len(iter(self.t_train_loader))))):

                    s_d, batch_y = unpacked1
                    t_d, _ = unpacked2
                    model_optim.zero_grad()
                    s_d = s_d.double().to(self.device)  # [B,window_size,D]
                    t_d = t_d.double().to(self.device)  # [B,window_size]
                    batch_y = batch_y.double().to(self.device)
                    s_features, s_out = self.model(s_d)
                    t_features, t_out = self.model(t_d)

                    if self.args.is_minmax:   # 训练的时候要拿source的output去和source的GT去比，因为认为场景下target是得不到GT的
                        # 只预测窗口内的最后一个rul
                        batch_y_norm = batch_y / self.max_life
                        loss1 = loss_criterion(s_out[:, -1], batch_y_norm)

                    else:
                        loss1 = loss_criterion(s_out[:, -1], batch_y)

                    if self.args.type == 1 or self.args.type == 0:
                        if self.args.type == 1:
                            s_domain = self.D2(s_features)
                            t_domain = self.D2(t_features)
                        else:
                            s_domain = self.D1(s_out)
                            t_domain = self.D1(t_out)
                        loss2 = advLoss(s_domain.squeeze(1), t_domain.squeeze(1), 'cuda')
                        loss = loss1 + 0.1*loss2

                        #Block all classifiers
                        #loss = loss1

                    elif self.args.type == 2:
                        s_domain_bkb = self.D2(s_features)
                        t_domain_bkb = self.D2(t_features)
                        s_domain_out = self.D1(s_out)
                        t_domain_out = self.D1(t_out)
                        if epoch >= 5:  # 迭代初期不引入混合loss
                            fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1), 'cuda')
                            out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), 'cuda')

                            loss = loss1 + 0.1 * fea_loss + 0.5 * out_loss
                        else:
                            loss = loss1

                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                end_time = time()
                epoch_time = end_time - start_time
                train_loss = np.average(train_loss)  # avgerage loss

                # validation process:
                vali_loss = self.validation(self.s_vali_loader, self.t_vali_loader, loss_criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format(
                    epoch + 1, train_steps, train_loss, vali_loss, epoch_time))

                # At the end of each epoch, Determine if we need to stop and adjust the learning rate

                early_stopping(vali_loss, self.model, self.save_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                learning_rate_adapter(model_optim, vali_loss)

            # 读取最优的参数
            check_point = torch.load(self.save_path  + '/' + 'best_checkpoint.pth')
            self.model.load_state_dict(check_point)

        # test:
        average_enc_loss, average_enc_overall_loss, overall_score = self.test(self.test_loader)
        print("CMAPSS: RMSE test performace of enc is: ", average_enc_loss, " of enc overall is: ",
                  average_enc_overall_loss, 'socre of'
                                            'enc', overall_score)

        log_path = './logs/DA_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'batch_size','best_last_RMSE', 'score','windowsize', 'source_d',
                           'target_d','type', 'train', 'savepath', 'resume',
                           'resumepath','info']]
            write_csv(log_path, table_head, 'w+')

        time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间

        resume_dir = self.checkpoint_dir if self.args.resume else None
        a_log = [{'dataset': save_path, 'model': self.args.model_name, 'time': time_now,
                  'LR': self.args.learning_rate,
                  'batch_size': self.args.batch_size,
                  'best_last_RMSE': average_enc_loss,
                  'score': overall_score, 'windowsize': self.args.input_length, 'source_d': self.args.source_domain,
                  'target_d': self.args.target_domain,
                  'type': self.args.type,
                  'train': self.args.train,
                  'savepath': self.save_path,
                  'resume': self.args.resume,
                  'resumepath': resume_dir,
                  'info': self.args.info}]
        write_csv_dict(log_path, a_log, 'a+')

    # ---------------------------------- validation function -----------------------------------------
    def validation(self, s_vali_loader, t_vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for _, (unpacked1, unpacked2) in enumerate(tqdm(zip(self.s_vali_loader, self.t_vali_loader), total=min(len(iter(self.s_vali_loader)), len(iter(self.t_vali_loader))))):

            s_d, batch_y = unpacked1
            t_d, _ = unpacked2

            s_d = s_d.double().to(self.device)  # [B,window_size,D]
            t_d = t_d.double().to(self.device)  # [B,window_size]
            batch_y = batch_y.double().to(self.device)

            s_features, s_out = self.model(s_d)
            t_features, t_out = self.model(t_d)

            if self.args.is_minmax:
                batch_y = batch_y / self.max_life
                loss = criterion(s_out[:, -1], batch_y)
            else:
                loss = criterion(t_out[:, -1], batch_y)

            total_loss.append(loss.item())

        average_vali_loss = np.average(total_loss)

        self.model.train()
        return average_vali_loss

    # ----------------------------------- test function ------------------------------------------
    def test(self, test_loader):
        self.model.eval()
        enc_pred = []
        gt = []
        for i, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.double().double().to(self.device)
            batch_y = batch_y.double().double().to(self.device)

            if self.args.is_minmax:
                t_features, t_out = self.model(batch_x)
                outputs = t_out * self.max_life

            else:
                t_features, t_out = self.model(batch_x)  # outputs[B,window_size]
                outputs = t_out

            batch_y = batch_y.detach().cpu().numpy()
            enc = outputs[:, -1].detach().cpu().numpy()

            gt.append(batch_y)
            enc_pred.append(enc)

        gt = np.concatenate(gt).reshape(-1, 1)
        enc_pred = np.concatenate(enc_pred).reshape(-1, 1)

        if self.args.save_test:
            result_path = self.save_path + '/result.npz'
            np.savez(result_path, test_preds=enc_pred, test_trues=gt,)

        # 算的就是RMSE
        average_enc_loss = np.sqrt(mean_squared_error(enc_pred, gt))
        average_enc_overall_loss = np.sqrt(mean_squared_error(enc_pred, gt))

        # 计算score
        overall_score = self.score_compute(enc_pred, gt)

        return average_enc_loss, average_enc_overall_loss, overall_score

    def score_compute(self, pred, gt):
        # pred [B] gt[B]
        score_list = np.where(pred - gt < 0, np.exp(-(pred - gt) / 13) - 1, np.exp((pred - gt) / 10) - 1)

        # 这里有的paper求均值，有的求和。实验里面先全都求和计算score
        score = np.sum(score_list)

        return score





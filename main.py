# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# %%
import warnings

warnings.filterwarnings("ignore")
# %%
from Experiment.Experiment import Exp
from Experiment.Experiment_DA import Exp_DA
import torch


# %%

"""
dataset_name: choose the dataset we used now

MAXLIFE_CMAOPSS: the knee point of RUL in each engine's RUL --- for CMAPSS

n_heads_full: the head num of FullAttention, used for ablation study
n_heads_log: the head num of LogSparseAttention
n_heads_local: the head num of LocalAttention
n_heads_prob: the head num of ProbSparseAttention
n_heads_FFT: the head num of FFT
n_heads_auto: the head num of AutoCorrelation

moving_avg: the kernel_size for decomposition block
enc_layer_num : the number of layers in Encoder
predictor_type: choose which Predictor used

learning_rate_patience: learning rate change after learning_rate_patience's epoch
learning_rate_factor: the percentage of learning rate change
arly_stop_patience: the time of training stop, when vali loss always bigger

enc_pred_loss: the loss function we choosed
sigma_faktor: The larger the value, the smaller the simga and the narrower the distribution of weights
anteil: The larger the value, the larger the subsequent distribution
smooth_loss: use the smooth_loss or not
"""



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False




if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--model_name', default='AIGCN', type=str,
                        help='[LeNet, LSTM, Transformer,Autoformer, PatchTST, AGCNN, Dual_Mixer, IMDSSN '
                             'Transformer_domain,FCSTGNN, AIGCN]')

    parser.add_argument('--train', default=True, type=str2bool,
                        help='Train or test')

    parser.add_argument('--resume', default=False, type=str2bool,
                        help='load checkpoint or not')

    parser.add_argument('--save_test', default=True, type=str2bool,
                        help='Save pred and target when testing')

    parser.add_argument('--save_path', default='exp0',
                        help='If path is None, exp_id add 1 automatically:if train, it wiil be useful')

    parser.add_argument('--resume_path', default='exp67',
                        help='if resume is True, it will be useful')

    parser.add_argument('--train_epochs', default=300, type=int,
                        help='train_epochs')

    parser.add_argument('--learning_rate', default=0.005, type=float,
                        help='lr')

    parser.add_argument('--info', default='main test', type=str,
                        help='extra information')

    # 1. load data parameter - common
    parser.add_argument('--dataset_name', default='CMAPSS', type=str,
                        help='[CMAPSS,N_CMAPSS,XJTU]')

    parser.add_argument('--Data_id_CMAPSS', default="FD003", type=str,
                        help='for CMAPSS')

    parser.add_argument('--Data_id_N_CMAPSS', default="", type=str,
                        help='for N_CMAPSS')

    parser.add_argument('--input_length', default=60, type=int,
                        help='input_lenth')

    parser.add_argument('--validation', default=0.1, type=float,
                        help='validation')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='bs')
    # AATE
    # parser.add_argument('--tuned_hyperparam', default=True, type=bool, help='if true it will use tune the hyperparameters datasetspecfic')
    parser.add_argument('--ablation_mode', default='mean', type=str, help='[identity, fixed, zero, CDI, staticAdj]')
    parser.add_argument('--AATE_dim', default = 10, type=int, help='AATE dimension')
    parser.add_argument( '--seed', type=int, default=1, help="Random seed" )

    # 1.1. load data parameter
    # CMAPSS
    parser.add_argument('--MAXLIFE_CMAPSS', default=125, type=int,
                        help='maxlife for cmapss')

    parser.add_argument('--normalization_CMAPSS', default="minmax", type=str,
                        help='way for norm')

    # N_CMAPSS
    parser.add_argument('-s', type=int, default=1, help='stride of window')
    parser.add_argument('--sampling', type=int, default=10,
                        help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('--change_len', type=str2bool, default=True,
                        help='re-generate data when you change input_len')
    parser.add_argument('--rate', type=float, default=0.8, help='max_life related')

    # 2. model parameter -common
    parser.add_argument('--d_model', default=64, type=int,
                        help='embedding dimension ')

    parser.add_argument('--d_ff', default=128, type=int,
                        help='embedding dimension ')

    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout')

    parser.add_argument('--patch_size', type=int, default=5, help='patch')

    # PV2 parameter
    parser.add_argument('--lamda', default=0.1, type=float)

    # Dlnear parameter
    parser.add_argument('--kernel_size', default=3, type=int,
                        help='kernel for Dlinear')

    parser.add_argument('--stride', default=1, type=int,
                        help='stride for Dlinear')

    # diff prediction but it is not correct ,so set it False
    parser.add_argument('--is_diff', default=False, type=str2bool,help='是否进行差分')

    parser.add_argument('--is_minmax', default=True, type=str2bool,help='是否进行最大最小标准化')

    # cross_seg
    parser.add_argument('--n_seg', type=int, default=8, help='seg for cross attention ')

    # Nbeats
    parser.add_argument('--n_stacks', type=int, default=2, help='n for stacks ')


    # Transformer
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads, for patchTST d_model should be divided by n_heads')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # patchTST
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--patch_stride', type=int, default=4, help='stride')

    # autoformer
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # DA_Transformer
    parser.add_argument('--DA', default=False, type=str2bool, help='Domain adaptation ')
    parser.add_argument('--source_domain', default="FD001", type=str,
                        help='source')
    parser.add_argument('--target_domain', default="FD001", type=str,
                        help='target')
    parser.add_argument('--type', type=int, default=2, help='loss type ')

    # Classify
    parser.add_argument('--Classify', default=False, type=str2bool, help='Domain adaptation ')
    parser.add_argument('--D1_lr', default=0.0001, type=float,
                        help='lr')


    args = parser.parse_args()

    print(f"|{'=' * 101}|")
    # print all configs
    # 使用__dict__方法获取参数字典，之后遍历字典
    for key, value in args.__dict__.items():
        # turn all the config to str
        # 因为参数不一定都是str，需要将所有的数据转成str
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")




    if args.dataset_name == 'CMAPSS':
        # data path
        args.data_path_CMAPSS = "./CMAPSS"
        args.difference_CMAPSS = False

        save_path = args.Data_id_CMAPSS

    elif args.dataset_name == 'N_CMAPSS':

        save_path = args.Data_id_N_CMAPSS

    elif args.dataset_name == 'XJTU':

        save_path = 'XJTU'

    else:
        raise ValueError('without corresponding dataset')

    args.use_gpu = True  # True if torch.cuda.is_available() else False
    args.gpu = 0

    args.optimizer = "Adam"
    args.learning_rate_patience = 10
    args.learning_rate_factor = 0.3
    args.early_stop_patience = 10


    args.enc_pred_loss = "MSE"  # "MSE"
    args.smooth_loss = None  # None
    if args.DA == True:
        exp = Exp_DA(args)
        exp.train(save_path)
    else:
        exp = Exp(args)
        exp.start()



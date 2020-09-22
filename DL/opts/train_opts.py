from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that run_infer_opts.py resides in

class train_opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 training options")
        self.parser.add_argument("--num_epochs", type=int, help="number of epochs", default=20)
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=16)  #
        self.parser.add_argument("--weights_save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=10)

        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["dense2value",
                                          'dense2value_lite'
                                          ],
                                 default="dense2value"
                                 )
        self.parser.add_argument('--train_files',
                                 default='150train.txt',

        )
        self.parser.add_argument('--val_files',default='150val.txt')

        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 #default='/home/roit/models/monodepth2/checkpoints/05-28-04:30/models/weights_19',
                                 )
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/home/roit/datasets/npmcm2020/airport_150'
                                 #default = '/home/roit/datasets/Binjiang/'
                                 #default="/home/roit/datasets/MC"
                                 #default = "/home/roit/datasets/VisDrone2"
                                )

        self.parser.add_argument("--gt_path",default="/home/roit/datasets/npmcm2020/AMOS/AMOS20200313/airports.csv")



        self.parser.add_argument('--lambdas',default=[1,1,0.5])
        self.parser.add_argument('--columns',default=['VIS1K','MOR_RAW','timestamp'])
        self.parser.add_argument('--name_loss',default = {'total':0,'VIS1K':0,'MOR_RAW':0,'timestamp':0})
        self.parser.add_argument('--name_metric',default = {'abs_rel':[0,0,0],
                                                            'abs_log_rel':[0,0,0]
                                                            }
                                 )

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 #default='/home/roit/models/monodepth2/eval_test'
                                 default='/home/roit/models/npmcm2020'
                                 #default = '/home/roit/models/monodepth2/visdrone'
                                  )
        self.parser.add_argument('--posecnn',default='en_decoder',choices=['en_decoder','share-encoder','posecnn'])



        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default='foggy',
                                 choices=["kitti",
                                          "kitti_odom",
                                          "kitti_depth",
                                          "mc",
                                          'visdrone',
                                          'custom_mono'])


        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--height",type=int,help="model input image height",
                                 default=384#dense2value
                                 )
        self.parser.add_argument("--width",type=int,help="model input image width",
                                 default=640#dense2value
                                 )



        self.parser.add_argument("--scales",nargs="+",type=int,help="scales used in the loss",default=[0])


        # OPTIMIZATION options

        self.parser.add_argument("--learning_rate",type=float,help="learning rate",default=1e-4)
        self.parser.add_argument("--start_epoch",type=int,help="for subsequent training",
                                 #default=10,
                                 default=0,

                                 )

        self.parser.add_argument("--scheduler_step_size",type=int,help="step size of the scheduler",default=15)

        # LOADING args for subsquent training or train from pretrained/scratch

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load, for training or test",
                                 default=["encoder",
                                          "depth",
                                          "pose_encoder",
                                          "pose"
                                          #"posecnn"
                                          ])

        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch or subsequent training from last",
                                 default="scratch",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--encoder_path",
                                 type=str,
                                 help="pretrained from here",
                                 #default=None,

                                 default="/home/roit/models/torchvision/official/resnet18-5c106cde.pth",
                                 )
        self.parser.add_argument('--net_arch', default='en_decoder',
                                 choices=['en_decoder', 'share-encoder', 'cnn'])
        self.parser.add_argument("--posecnn_path",
                                 type=str,
                                 help="pretrained from here",
                                 #default="/home/roit/models/SCBian_official/k_pose.tar",
                                 default=None,

                                 )




        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=7)



        # LOGGING options
        self.parser.add_argument("--tb_log_frequency",
                                 type=int,
                                 help="number of batches(step) between each tensorboard log",
                                 default=10)

    def args(self):
        self.options = self.parser.parse_args()
        return self.options

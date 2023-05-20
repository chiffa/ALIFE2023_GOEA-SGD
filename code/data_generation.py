import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.utils.spectral_norm as spectral_norm

import os
from copy import deepcopy
from datetime import datetime
import numpy as np
from scipy import stats as scst
import matplotlib.pyplot as plt
from csv import writer as csv_writer
import pickle
from tabulate import tabulate
from collections import OrderedDict
import secrets
from random import sample
import yaml
import string
from pathlib import Path

char_set = string.ascii_uppercase + string.digits


class SavePaths(object):
    """
    A class that allows a verifiable mapping of saved paths
    """

    def __init__(self):
        root_reg = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
        self.top_path = Path(__file__).parent / 'runs' / root_reg
        self.top_path.mkdir(parents=True)

        self.important_params = Path(self.top_path) / 'training_params.yaml'
        self.param_counts = Path(self.top_path) / 'model_params.txt'


        self.train_trace_tsv_suffix = 'train_trace.tsv'
        self.flat_sweep_tsv_suffix = 'flat_sweep_combined.tsv'
        self.sweeps_map_suffix = 'Sweeps_map.png'
        self.sweeps_summary_suffix = 'Sweeps_summary.png'
        self.go_ea_log_suffix = 'GO_EA_log.tsv'
        self.gener_vs_redun_suffix = 'generalization_vs_redundancy.yaml'
        self.circular_sweep_suffix = 'circular_sweep.tsv'
        self.sgd_angle = 'sgd_angle.tsv'


        self.image_sample = Path(self.top_path) / 'last_render.png'
        self.training_trace_analysis = Path(self.top_path) / 'training_traces.png'

    def conjugate(self, net_instance, suffix):

        return net_instance.modulate_path(self.top_path, suffix)


active_savepath = SavePaths()


########################################################################
#
# Edit values in Environment __init__ to change experimental conditions
#
########################################################################


class Environment(object):
    """
    A class that wraps the parameter used to build the network and provide
    parameters used to train it in a robust manner
    """

    def __init__(self):
        self.image_folder = "./data"
        self.image_size = 28

        # self.classifier_latent_maps = 16  # stable
        # self.classifier_latent_maps = 4  # brittle
        self.classifier_latent_maps = 8  # evo

        self.train_batch_size = 32  # default
        # self.train_batch_size = 4  # stable
        # self.train_batch_size = 128  # brittle
        # self.train_batch_size = 1024  # evo

        self.test_batch_size = 1024
        self.epochs = 10

        # self.source_image_dropout = 0.1  # stable
        # self.dropout = 0.25  # stable
        # self.dropout = 0.1  # default
        self.source_image_dropout = 0.0  # brittle
        self.dropout = 0.0  # brittle

        self.learning_rate = 0.002
        self.momentum = 0.0

        # self.linear_width = 48  # stable
        self.linear_width = 24  # evo / default
        # self.linear_width = 12  # brittle

        self.eval_mode = False  # if absent, True: quakes are done with drop-out
        self.annotation = 6
        # -1 for sgd param sweep
        # 1 for GO-EA
        # 2 for rob/gen
        # 3 for fine-tuning,
        # 4 for GO-EA parameter sweep
        # 5 for GO-EA vector analysis
        # 6 for SGD batch-size update angle divergence
        self.fine_tune_epochs = -1  # back to -1 for GO-EA

    def todict(self) -> dict:
        """
        Converts its parameters to a python dict

        :return:
        """
        retdict = {
            'image_folder': self.image_folder,
            'image_size': self.image_size,
            'classifier_latent_maps': self.classifier_latent_maps,
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'epochs': self.epochs,
            'source_image_dropout': self.source_image_dropout,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'linear_width': self.linear_width,
            'annotation': self.annotation,
            'fine_tune_epochs': self.fine_tune_epochs,
        }

        return retdict

    def fromdict(self, retdict: dict) -> None:
        """
        Initializes itself from a dictionary containing its own parameters

        :param retdict: dict to load parameters from
        :return:
        """
        self.image_folder = retdict['image_folder']
        self.image_size = retdict['image_size']
        self.classifier_latent_maps = retdict['classifier_latent_maps']
        self.train_batch_size = retdict['train_batch_size']
        self.test_batch_size = retdict['test_batch_size']
        self.epochs = retdict['epochs']
        self.source_image_dropout = retdict['source_image_dropout']
        self.dropout = retdict['dropout']
        self.learning_rate = retdict['learning_rate']
        self.momentum = retdict['momentum']
        self.linear_width = retdict['linear_width']
        self.annotation = retdict['annotation']
        self.fine_tune_epochs = retdict['fine_tune_epochs']

    def toyaml(self, file_path: Path = active_savepath.important_params) -> None:
        """
        Dumps its own parameters to a yaml file

        :param file_path: path where to store the .yaml parameters dump
        :return:
        """
        dumpdict = self.todict()
        with open(file_path, 'w') as outfile:
            yaml.dump(dumpdict, outfile, default_flow_style=False)

    def fromyaml(self, file_path: Path):
        """
        Recovers its own parameters from a yaml file

        :param file_path: path from where to retrieve the .yaml paramaeters dump
        :return:
        """
        with open(file_path, 'r') as infile:
            dumpdict = yaml.load(infile)
        self.fromdict(dumpdict)


active_environment = Environment()  # create the environment
active_environment.toyaml()  # save the environment that will be used for the run

# set up the transforms to convert to the expected dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                            (0.5, 0.5, 0.5))])

gw_transform = transforms.Compose([
                               transforms.Resize(active_environment.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,),
                                                    (0.5,)),])

# set the dataset to be used to MNIST
active_dataset = torchvision.datasets.MNIST


def generate_oblvious_loaders(oblivious_to=()):
    """
    This routine generates dataloaders that do not know about numbers listed in "oblivious to"
    list/tuple. It is used to study augmentative learning.

    :param oblivious_to: Numbers that the loader will not differentiate
    :return:
    """
    _oblivious_train_ds = active_dataset(root=active_environment.image_folder,
                          train=True,
                          download=True,
                          transform=gw_transform)

    _oblivious_test_ds = active_dataset(root=active_environment.image_folder,
                             train=False,
                             download=True,
                             transform=gw_transform)

    for oblivioned in oblivious_to:
        _oblivious_train_ds.targets[_oblivious_train_ds.targets == oblivioned] = -1
        _oblivious_test_ds.targets[_oblivious_test_ds.targets == oblivioned] = -1

    _oblivious_train_ds_loader = torch.utils.data.DataLoader(_oblivious_train_ds,
                                                  batch_size=active_environment.train_batch_size,
                                                  shuffle=True,
                                                  num_workers=2)

    _oblivious_test_ds_loader = torch.utils.data.DataLoader(_oblivious_test_ds,
                                                 batch_size=active_environment.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=2)

    return _oblivious_train_ds_loader, _oblivious_test_ds_loader


# create the default and oblivious dataloaders
train_ds_loader, test_ds_loader = generate_oblvious_loaders()
oblivious_train_ds_loader, oblivious_test_ds_loader = generate_oblvious_loaders([8, 9])


def show_image(img):
    """
    A function to render an image and save it to disk for further inspection. Mainly used for
    debugging

    :param img:
    :return:
    """
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(active_savepath.image_sample)
    plt.clf()


def timeit(start_time=[datetime.now()]):
    """
    A function used to measure wall clock time since the start of script execultion. It uses a
    clojure pattern to remember time at which the function was defined (hence script started)

    :param start_time: list containing a datetime object, considered as a starting point. By
    default, the time when the function started to be executed.
    :return:
    """
    ret_delta = datetime.now() - start_time[0]
    start_time[0] = datetime.now()
    return ret_delta


# dataiter = iter(train_ds_loader)
# images, labels = dataiter.next()
# show_image(torchvision.utils.make_grid(images))

# requests confirmation to resume execution from a previous model, if one is being loaded
print("executing %s, modified last on %s" % (__file__,
                                             datetime.fromtimestamp(os.path.getmtime(
                                                 __file__)).isoformat()))
input("press enter to continue")


class PrintLayer(nn.Module):
    """
    A class to print layers of ANNs, mostly for debugging purposes
    """

    def __init__(self, location):
        super(PrintLayer, self).__init__()
        self.location = location

    def forward(self, x):
        # Do your print / debug stuff here
        print("nn.debug, layer %s " % self.location, x.shape)
        return x



class SeparableConv2d(nn.Module):

    def __init__(self, nin, nout, kernel_size=3, padding=0):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(nin, nin,
                                   kernel_size=kernel_size, padding=padding, groups=nin)

        self.pointwise = nn.Conv2d(nin, nout,
                                   kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Net(nn.Module):
    """
    Defines the base class for the networks
    """

    def __init__(self):

        super().__init__()

        self.classifier_latent_maps = 32
        self.ngpu = 1

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.classifier_latent_maps,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.classifier_latent_maps,
                      out_channels=self.classifier_latent_maps*2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=self.classifier_latent_maps*2,
                      out_channels=self.classifier_latent_maps*4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.classifier_latent_maps*4,
                      out_channels=self.classifier_latent_maps*4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=self.classifier_latent_maps*4,
                      out_channels=self.classifier_latent_maps*8,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.classifier_latent_maps*8,
                      out_channels=self.classifier_latent_maps*8,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(self.classifier_latent_maps*8*64, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 10))


    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output


class ShortNet(nn.Module):
    """
    Defines  a base class for networks with less lalyers
    """

    def __init__(self):

        super().__init__()

        self.classifier_latent_maps = 16
        self.ngpu = 1

        self.main = nn.Sequential(

            # 28x28x1
            # PrintLayer(0),
            nn.Conv2d(in_channels=1,
                      out_channels=self.classifier_latent_maps,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            # PrintLayer(1),
            nn.MaxPool2d(2, 2),  # 14x14x16
            # PrintLayer(2),

            nn.Conv2d(in_channels=self.classifier_latent_maps,
                      out_channels=self.classifier_latent_maps*2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            # PrintLayer(3),
            nn.MaxPool2d(2, 2),  # 7x7x32
            # PrintLayer(4),

            nn.Flatten(),
            # PrintLayer(5),

            nn.Linear(self.classifier_latent_maps*2*7*7, 128),
            nn.ReLU(),

            nn.Linear(128, 10))

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output


class CompressedNet(nn.Module):
    """
    Defines the base class for the ANN with reasonable size, that is used in most experiments
    """

    def __init__(self):

        super().__init__()

        self.classifier_latent_maps = active_environment.classifier_latent_maps
        self.source_dropout = active_environment.source_image_dropout
        self.dropout = active_environment.dropout
        self.model_id = ''.join(sample(char_set * 10, 10))
        self.linear_width = active_environment.linear_width
        self.ngpu = 1

        self.main = nn.Sequential(

            # 28x28x1
            # PrintLayer(0),
            nn.Dropout(self.source_dropout),
            spectral_norm(nn.Conv2d(in_channels=1,
                      out_channels=self.classifier_latent_maps,
                      kernel_size=5,
                      stride=1,
                      padding=2)),
            nn.ReLU(),
            # PrintLayer(1),
            nn.MaxPool2d(2, 2),  # 14x14x8
            # PrintLayer(2),

            nn.Dropout(self.dropout),
            spectral_norm(nn.Conv2d(in_channels=self.classifier_latent_maps,
                      out_channels=self.classifier_latent_maps*2,
                      kernel_size=5,
                      stride=1,
                      padding=2)),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # 7x7x16

            nn.Dropout(self.dropout),
            spectral_norm(nn.Conv2d(in_channels=self.classifier_latent_maps*2,
                      out_channels=self.classifier_latent_maps,
                      kernel_size=3,
                      stride=1,
                      padding=1)),
            nn.ReLU(),

            # 7x7x8

            nn.Dropout(self.dropout),
            spectral_norm(nn.Conv2d(in_channels=self.classifier_latent_maps,
                      out_channels=int(self.classifier_latent_maps/2),
                      kernel_size=3,
                      stride=1,
                      padding=1)),
            nn.ReLU(),

            # 7x7x2

            # PrintLayer(3),
            # PrintLayer(4),

            nn.Flatten(),
            # PrintLayer(5),
            nn.Dropout(self.dropout),
            spectral_norm(nn.Linear(int(self.classifier_latent_maps/2)*7*7, self.linear_width)),
            nn.ReLU(),

            nn.Linear(self.linear_width, 10)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output

    def modulate_path(self, top_path, suffix):
        """
        Adjusts the path where the model will be saved

        :param top_path:
        :param suffix:
        :return:
        """
        path_complement = type(self).__name__ + '_' + self.model_id + '_' + suffix
        modpath = Path(top_path) / path_complement
        return modpath


    def archive(self, path=active_savepath.top_path):
        """
        Saves the model for future use in a .dmp file within a run

        :param path: path where to save the model
        :return:
        """

        path_complement = type(self).__name__ + '_' + self.model_id + '.dmp'
        savepath = Path(path) / path_complement

        with open(savepath, 'wb') as outfile:
            savedict = {
                'classifier_latent_maps': self.classifier_latent_maps,
                'dropout': self.dropout,
                'ngpu': self.ngpu,
                'linear_width': self.linear_width,
                'state_dict': self.main.state_dict()
            }
            print('archiving the net to %s' % savepath)
            pickle.dump(savedict, outfile)

    def resurrect(self, path):
        """
        Recovers the model from the a

        :param path:
        :return:
        """
        self.model_id = Path(path).stem.split('_')[1]

        with open(path, 'rb') as infile:
            savedict = pickle.load(infile)
            self.classifier_latent_maps = savedict['classifier_latent_maps']
            self.dropout = savedict['dropout']
            self.ngpu = savedict['ngpu']
            self.linear_width = savedict['linear_width']
            self.main.load_state_dict(savedict['state_dict'])
            print('resurrecting the net to %s' % path)

    def fine_tuned_id(self):
        """
        In case of ine-tuning, give a random ID to the network before saving it

        :return:
        """
        self.model_id += '_ft_' + ''.join(sample(char_set * 4, 4))


class MinimalNet(nn.Module):
    """
    A class that defines an ANN with as small of a size as it is possible given the data size
    """

    def __init__(self):

        super().__init__()

        self.classifier_latent_maps = 8
        self.dropout = .1
        self.ngpu = 1

        self.main = nn.Sequential(
            # 28x28x1
            spectral_norm(
                nn.Conv2d(in_channels=1,
                          out_channels=self.classifier_latent_maps,
                          kernel_size=3,
                          stride=1,
                          padding=0)),
            nn.ReLU(),
            # 26x26x8
            nn.MaxPool2d(2, 2),
            # 13x13x8
            nn.Dropout(self.dropout),

            SeparableConv2d(8, 26),
            # 11x11x26
            nn.ReLU(),

            nn.Dropout(self.dropout),

            SeparableConv2d(8, 26, padding=1),
            # 11x11x26
            nn.ReLU(),

            nn.Dropout(self.dropout),

            nn.AvgPool2d(26),
            # 26

            spectral_norm(
                nn.Linear(26, 16)),
            #16

            nn.ReLU(),

            spectral_norm(
                nn.Linear(16, 10)),
            #10

            nn.Softmax()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output


class NormShortNet(nn.Module):
    """
    Defines a reference short net
    """

    def __init__(self):

        super().__init__()

        self.classifier_latent_maps = 16
        self.dropout = .25
        self.ngpu = 1

        self.main = nn.Sequential(

            # 28x28x1
            # PrintLayer(0),
            nn.Dropout(self.dropout),
            spectral_norm(nn.Conv2d(in_channels=1,
                      out_channels=self.classifier_latent_maps,
                      kernel_size=5,
                      stride=2,
                      padding=2)),
            nn.ReLU(),
            # PrintLayer(1),
            # nn.MaxPool2d(2, 2),  # 14x14x16
            # PrintLayer(2),

            nn.Dropout(self.dropout),
            spectral_norm(nn.Conv2d(in_channels=self.classifier_latent_maps,
                      out_channels=self.classifier_latent_maps*2,
                      kernel_size=5,
                      stride=2,
                      padding=2)),
            nn.ReLU(),

            # PrintLayer(3),
            # nn.MaxPool2d(2, 2),  # 7x7x32
            # PrintLayer(4),

            nn.Flatten(),
            # PrintLayer(5),
            nn.Dropout(self.dropout),
            spectral_norm(nn.Linear(self.classifier_latent_maps*2*7*7, 128)),
            nn.ReLU(),

            nn.Linear(128, 10))

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output


def count_parameters(model, dump_location=active_savepath.param_counts):
    """
    This function counts the number of model parameters for each layers, as well as the total
    parameter number and saves it in a file for future inspection

    :param model: model for which to count parameters
    :param dump_location: location where to safe the parameter counts
    :return:
    """
    table = [["Modules", "Parameters"]]
    total_params = 0

    for name, parameter in model.main.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.append([name, param])
        total_params += param
    table.append(['Total params', total_params])

    print(tabulate(table, headers="firstrow"))

    if not dump_location is None:
        with open(dump_location, 'w') as outfile:
            outfile.write(type(model).__name__+'/n')
            outfile.write(tabulate(table, headers='firstrow'))

    return total_params


def train_loop(num_epoch=10,
               net_class=ShortNet,
               save_path=active_savepath,
               explicit_net=None,
               explicit_train_loader=train_ds_loader,
               explicit_test_loader=test_ds_loader):
    """
    This function trains the ANN model.

    :param num_epoch: epoch to train the model for
    :param net_class: network class that will be used
    :param save_path: where to save the model
    :param explicit_net: existing model to start the training from, if any
    :param explicit_train_loader: existing dataloader for training
    :param explicit_test_loader: existing test dataloader
    :return:
    """

    if explicit_net is None:
        active_net = net_class()
        active_net.main.cuda()

    else:
        active_net = explicit_net
        active_net.main.cuda()

    # Switch over the comments for for fine-tuning
    # makes the optimizer oblivious to the occluded categories

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.SGD(active_net.main.parameters(),
                          lr=active_environment.learning_rate,
                          momentum=active_environment.momentum)

    training_trace = []

    print("epoch\ttr_acc\tvl_acc\ttr_loss\tvl_loss\ttime_to_train")

    timeit()

    for epoch in range(1, num_epoch+1):
      train_loss = 0.0
      valid_loss = 0.0

      train_correct = 0.0
      valid_correct = 0.0


      active_net.main.train()

      # loss_log = []

      for img, lbl in explicit_train_loader:
        img = img.cuda()
        lbl = lbl.cuda()

        optimizer.zero_grad()
        predict = active_net.forward(img)
        loss = loss_fn(predict, lbl)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * img.size(0)
        # loss_log.append(loss.item() * img.size(0))
        train_correct += (torch.argmax(predict, dim=1) == lbl).float().sum()


      active_net.main.eval()  # Dropout gets disabled here

      for img, lbl in explicit_test_loader:
        img = img.cuda()
        lbl = lbl.cuda()

        predict = active_net.forward(img)
        # predict = active_net.main(img)
        loss = loss_fn(predict, lbl)

        valid_loss += loss.item() * img.size(0)

        valid_correct += (torch.argmax(predict, dim=1) == lbl).float().sum()

      train_loss = train_loss / len(explicit_train_loader.sampler)
      valid_loss = valid_loss / len(explicit_test_loader.sampler)

      # print(loss_log)
      # loss_log = np.array(loss_log) / len(train_ds_loader.sampler)
      # print(loss_log)
      # loss_diff = loss_log[1:] - loss_log[:-1]
      # print(loss_diff)
      # av_loss = np.mean(loss_diff)

      train_accuracy = 100 * float(train_correct) / len(explicit_train_loader.sampler)
      valid_accuracy = 100 * float(valid_correct) / len(explicit_test_loader.sampler)

      training_trace.append([train_accuracy, valid_accuracy, train_loss, valid_loss])

      print("%s\t%.2f\t%.2f\t%.4f\t%.4f\t%s" % (epoch,
                                                train_accuracy, valid_accuracy,
                                                train_loss, valid_loss,
                                                str(timeit())))

    if not save_path is None:
        full_path = save_path.conjugate(active_net, save_path.train_trace_tsv_suffix)
        with open(full_path, 'w') as outfile:
            writer = csv_writer(outfile, delimiter='\t')
            writer.writerows(training_trace)

        active_net.archive()

    return active_net,  training_trace


def sweep_batch_updates(net_class=ShortNet,
                        explicit_net=None,
                        explicit_train_loader=train_ds_loader):
    """
    Performs the evaluation of update vectors for each batch and

    :param net_class: network class that will be used
    :param explicit_net: existing model to start the training from, if any
    :param explicit_train_loader: existing dataloader for training
    :return:
    """

    if explicit_net is None:
        active_net = net_class()
        active_net.main.cuda()

    else:
        active_net = explicit_net
        active_net.main.cuda()

    # Switch over the comments for for fine-tuning
    # makes the optimizer oblivious to the occluded categories

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.SGD(active_net.main.parameters(),
                          lr=active_environment.learning_rate,
                          momentum=active_environment.momentum)

    training_trace = []

    print("epoch\ttr_acc\tvl_acc\ttr_loss\tvl_loss\ttime_to_train")

    timeit()

    train_loss = 0.0
    valid_loss = 0.0

    train_correct = 0.0
    valid_correct = 0.0

    active_net.main.train()

    # loss_log = []

    linearized_tensor_list = []

    for img, lbl in explicit_train_loader:
        img = img.cuda()
        lbl = lbl.cuda()

        optimizer.zero_grad()

        base_state_dict = deepcopy(active_net.main.state_dict())
        predict = active_net.forward(img)
        loss = loss_fn(predict, lbl)
        loss.backward()
        optimizer.step()  # basically, at this step

        updated_state_dict = deepcopy(active_net.main.state_dict())
        active_net.main.load_state_dict(base_state_dict)


        linearized_tensor_aggregator = []

        for (_name, _tensor), (u_name, u_tensor) in zip(base_state_dict.items(),
                                                       updated_state_dict.items()):
            if _name != u_name:
                raise Exception('Different layer architectures')

            linearized_tensor_aggregator += deepcopy(u_tensor - _tensor).flatten().tolist()

        linearized_tensor_list.append(linearized_tensor_aggregator)

    linearized_tensor_list = np.array(linearized_tensor_list)

    angle_matrix = linearized_tensor_list @ linearized_tensor_list.T
    sqrt_diag = np.sqrt(np.diag(angle_matrix))
    angle_matrix = angle_matrix / sqrt_diag[:, np.newaxis]
    angle_matrix = angle_matrix / sqrt_diag[np.newaxis:, ]
    angle_matrix = np.arccos(np.clip(angle_matrix, -1.0, 1.0)) * 180 / np.pi

    norm_list = np.linalg.norm(linearized_tensor_list, axis=1)

    return norm_list, angle_matrix


def save_and_render(training_trace_list, annotation=''):
    """
    Saves the training trace as well well as its graphic representation

    :param training_trace_list: list of training trace
    :param annotation: additional annotation, if needed
    :return:
    """
    pickle.dump(training_trace_list, open('train_trace.dmp', 'bw'))

    normalized_trace_list = []
    normalized_trace = []
    for trace in training_trace_list:
        for line in trace:
            normalized_line = [float(elt) for elt in line]
            normalized_trace.append(normalized_line)
        normalized_trace_list.append(normalized_trace)
        normalized_trace = []

    trace_arr = np.array(normalized_trace_list)
    print(trace_arr.shape)

    x = range(0, len(trace_arr[0, :, 0]))

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Training layout:\n%s' % annotation)

    ax1.errorbar(x, np.mean(trace_arr[:, :, 0], axis=0),
                 scst.sem(trace_arr[:, :, 0]), fmt='.k', label='train_acc')
    ax1.errorbar(x, np.mean(trace_arr[:, :, 1], axis=0),
                 scst.sem(trace_arr[:, :, 1]), fmt='.r', label='valid_acc')
    ax1.legend()

    ax2.set_yscale("log")
    ax2.errorbar(x, np.mean(trace_arr[:, :, 2], axis=0),
                 scst.sem(trace_arr[:, :, 2]), fmt='.k', label='train_loss')
    ax2.errorbar(x, np.mean(trace_arr[:, :, 3], axis=0),
                 scst.sem(trace_arr[:, :, 3]), fmt='.r', label='valid_loss')

    ax2.legend()

    plt.savefig(active_savepath.training_trace_analysis)
    plt.clf()


def test(active_net,
         eval_mode=True,
         explicit_dataloader=test_ds_loader):
    """
    Calculates the loss and accuracy on the test data

    :param active_net:
    :param eval_mode:
    :param explicit_dataloader:
    :return:
    """

    valid_loss = 0.0
    valid_correct = 0.0

    # Switch over the comments for for fine-tuning
    # makes the optimizer oblivious to the occluded categories

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    if not eval_mode:
        active_net.main.train()  # enables dropout

    for img, lbl in explicit_dataloader:
        img = img.cuda()
        lbl = lbl.cuda()

        predict = active_net.forward(img)
        # predict = active_net.main(img)
        loss = loss_fn(predict, lbl)
        valid_loss += loss.item() * img.size(0)

        valid_correct += (torch.argmax(predict, dim=1) == lbl).float().sum()

    valid_loss = valid_loss / len(test_ds_loader.sampler)
    valid_accuracy = 100 * valid_correct / len(test_ds_loader.sampler)

    if eval_mode:
        active_net.main.eval()

    return valid_loss, float(valid_accuracy)


def train_test(active_net):
    """
    Calculates the loss and accuracy on the training data (dependency of GO-EA search)

    :param active_net: the ANN model for which we are computing it.
    :return:
    """
    train_loss = 0.0
    train_correct = 0.0

    loss_fn = nn.CrossEntropyLoss()

    for img, lbl in train_ds_loader:
        img = img.cuda()
        lbl = lbl.cuda()

        predict = active_net.forward(img)
        # predict = active_net.main(img)
        loss = loss_fn(predict, lbl)
        train_loss += loss.item() * img.size(0)
        train_correct += (torch.argmax(predict, dim=1) == lbl).sum()

    train_loss = train_loss / len(train_ds_loader.sampler)
    test_accuracy = 100 * train_correct / len(train_ds_loader.sampler)

    return train_loss, float(test_accuracy)


def generate_perturbation(state_dict, diameter, manual_rand_seed=None):
    """
    Generates a perturbation to the tensor of weights of the ANN, including from a seed

    :param state_dict: state dict pf the ANN
    :param diameter: size of the perturbation
    :param manual_rand_seed: manually provided seed
    :return:
    """

    new_state_dict = OrderedDict()

    # debug_flag = True

    if not manual_rand_seed is None:
        if manual_rand_seed > -9223372036854775808 and manual_rand_seed < 18446744073709551615:
            torch.manual_seed(manual_rand_seed)
        else:
            raise Exception('random seed %s outside the range' % manual_rand_seed)

    for _name, _tensor in state_dict.items():
        tensor_av = torch.mean(torch.abs(_tensor))

        perturbation_tensor = torch.normal(0.0, tensor_av*diameter, size=_tensor.shape).cuda()

        # if debug_flag == True:
        #     print(perturbation_tensor)
        #     debug_flag = False

        new_state_dict[_name] = _tensor + perturbation_tensor

        if not manual_rand_seed is None:
            # Here we use a cryptographically secure seed to avoid getting interference with setting
            #   random seeds manually elsewhere
            new_seed = secrets.randbelow(int(0x8000_0000_0000_0000) +
                                         int(0xffff_ffff_ffff_ffff)) - int(0x8000_0000_0000_0000)
            torch.manual_seed(new_seed)

    return new_state_dict


def quake(active_net, diameter=.1, shakes=3, fixed_rand_seed=None,
          explicit_dataloader=test_ds_loader):
    """
    Generates several perturbation of weights of an ANN and evaluates the loss and accuracy
    of perturbed models. Is used for the evaluation of minima flatness

    :param active_net: the weights of ANN as a tensor dict
    :param diameter: by how much perturb the weights
    :param shakes: how many times to perform the perturbation
    :param fixed_rand_seed: potentially deterministic random seed
    :param explicit_dataloader: dataloader for the data to be used to evaluate the loss and accuracy
    :return:
    """

    # print('quaking with random seed %d' % fixed_rand_seed)

    base_state_dict = deepcopy(active_net.main.state_dict())
    quake_log = []

    if isinstance(diameter, list):
        diameter_list = np.linspace(diameter[0], diameter[1], num=shakes)
        if not fixed_rand_seed:
            fixed_rand_seed = secrets.randbelow(int(0x8000_0000_0000_0000) +
                                         int(0xffff_ffff_ffff_ffff)) - int(0x8000_0000_0000_0000)
            print('diameter sweep detected, fixing random seed % d' % fixed_rand_seed)

    else:  # assumed to be a float
        diameter_list = [diameter] * shakes

    for diam in diameter_list:
        nsd = generate_perturbation(base_state_dict, diam, fixed_rand_seed)
        active_net.main.load_state_dict(nsd)
        val_loss, val_acc = test(active_net, explicit_dataloader=explicit_dataloader)
        quake_log.append([diam, val_acc, val_loss])
        print("shake for diam %.2f:\tacc:%.4f\tloss:%.2f" % (diam, val_acc, val_loss))

    active_net.main.load_state_dict(base_state_dict)

    return quake_log


def generate_fixed_perturbation(state_dict, diameter, manual_rand_seed):
    """
    Generates a perturbation with a mandatory  fixed seed

    :param state_dict:
    :param diameter:
    :param manual_rand_seed:
    :return:
    """

    new_state_dict = OrderedDict()

    torch.manual_seed(manual_rand_seed)

    for _name, _tensor in state_dict.items():
        perturbation_tensor = torch.normal(0.0, diameter, size=_tensor.shape).cuda()
        new_state_dict[_name] = _tensor + perturbation_tensor

    return new_state_dict


def search_best_seed(active_net, diameter=.1, tries=100, enact_update=False):
    """
    Performs the search for the seed in the neighbourhood allowing to achieve the best loss/accuracy

    :param active_net: the ANN for which we are performing the optimization
    :param diameter: diameter in which to search
    :param tries: how many attempts to male before selecting the best
    :param enact_update: whether to update the model whenever a better one is found.
    :return:
    """

    # print('quaking with random seed %d' % fixed_rand_seed)

    base_state_dict = deepcopy(active_net.main.state_dict())

    best_seed = None
    train_loss, train_acc = train_test(active_net)

    best_loss = train_loss
    best_acc = train_acc

    print('starting search with loss %.4f and accuracy %.2f. last generation took %s' % (best_loss,
                                                                                best_acc, timeit()))
    # print('base_dict_snap:', base_state_dict['5.bias'])

    # try_seed = secrets.randbelow(int(0x8000_0000_0000_0000) +
    #                                       int(0xffff_ffff_ffff_ffff)) - int(0x8000_0000_0000_0000)
    previous_improvement_step = 0
    for i in range(tries):
        try_seed = secrets.randbelow(int(0x8000_0000_0000_0000) +
                                     int(0xffff_ffff_ffff_ffff)) - int(0x8000_0000_0000_0000)

        new_state_dict = generate_fixed_perturbation(base_state_dict, diameter, try_seed)
        active_net.main.load_state_dict(new_state_dict)
        train_loss, train_acc = train_test(active_net)

        print('testing seed #%d' % i, end="\r")
        # print('\tdebug loss %.4f; acc %.2f with seed %s' % (train_loss, train_acc, try_seed))
        # print(new_state_dict['5.bias'])
        if train_loss < best_loss:
            print('\timproved after %d steps: loss %.4f(%.4f); acc %.2f' % (
                i - previous_improvement_step, train_loss, train_loss - best_loss, train_acc))

            # print('test_dict_snap:', new_state_dict['5.bias'])
            previous_improvement_step = i
            best_seed = try_seed
            best_loss = train_loss
            best_acc = train_acc

    if enact_update and not best_seed is None:
        # print('branch1')
        new_state_dict = generate_fixed_perturbation(base_state_dict, diameter, best_seed)
        active_net.main.load_state_dict(new_state_dict)

    else:
        # print('branch2')
        active_net.main.load_state_dict(base_state_dict)

    resume_random = secrets.randbelow(int(0x8000_0000_0000_0000) +
                                      int(0xffff_ffff_ffff_ffff)) - int(0x8000_0000_0000_0000)

    torch.manual_seed(resume_random)

    # post_state_dict = deepcopy(active_net.main.state_dict())
    # print('post_dict_snap:', post_state_dict['5.bias'])

    return active_net, best_seed, best_loss, best_acc


def perform_circular_sweep(active_net, diameter=.1, tries=100):
    """
    Calculates the evaluation of loss and accuracy by doing a sweep and remembering the angle
    between the perturbation vectors.

    :param active_net: the ANN that is being evaluated
    :param diameter: diameter of perturbations
    :param tries: how many samples to evaluate
    :return:
    """

    active_net.main.cpu()

    base_state_dict = deepcopy(active_net.main.state_dict())
    active_net.main.cuda()
    train_loss, train_acc = train_test(active_net)

    print('starting sweep on %s with loss %.4f and accuracy %.2f.' % (
        active_net.model_id, train_loss, train_acc))

    loss_list = []
    acc_list = []
    linearized_tensor_list = []

    for i in range(tries):
        try_seed = secrets.randbelow(int(0x8000_0000_0000_0000) +
                                     int(0xffff_ffff_ffff_ffff)) - int(0x8000_0000_0000_0000)

        new_state_dict = OrderedDict()
        torch.manual_seed(try_seed)

        linearized_tensor_aggregator = []

        for _name, _tensor in base_state_dict.items():
            perturbation_tensor = torch.normal(0.0, diameter, size=_tensor.shape)
            linearized_tensor_aggregator += deepcopy(perturbation_tensor).numpy().flatten().tolist()
            # print('_tensor/perturbation are on cuda: %s/%s' % (_tensor.is_cuda,
            #                                                    perturbation_tensor.is_cuda))
            new_state_dict[_name] = _tensor + perturbation_tensor

        active_net.main.load_state_dict(new_state_dict)
        active_net.main.cuda()
        sweep_train_loss, sweep_train_acc = train_test(active_net)

        linearized_tensor_list.append(linearized_tensor_aggregator)
        loss_list.append(sweep_train_loss - train_loss)
        acc_list.append(sweep_train_acc - train_acc)

        print('testing seed #%d' % i, end="\r")


    active_net.main.load_state_dict(base_state_dict)

    resume_random = secrets.randbelow(int(0x8000_0000_0000_0000) +
                                      int(0xffff_ffff_ffff_ffff)) - int(0x8000_0000_0000_0000)

    torch.manual_seed(resume_random)

    linearized_tensor_list = np.array(linearized_tensor_list)
    loss_list = np.array(loss_list)
    acc_list = np.array(acc_list)

    angle_matrix = linearized_tensor_list @ linearized_tensor_list.T
    sqrt_diag = np.sqrt(np.diag(angle_matrix))
    angle_matrix = angle_matrix / sqrt_diag[:, np.newaxis]
    angle_matrix = angle_matrix / sqrt_diag[np.newaxis:, ]
    angle_matrix = np.arccos(np.clip(angle_matrix, -1.0, 1.0)) * 180 / np.pi

    return loss_list, acc_list, angle_matrix


def go_ea_train_loop(generations=200, edit_distance=0.1,
                     population=500, net_class=CompressedNet,
                     explicit_net=None):
    """
    The main loop to train the evolutionary GO-EA


    :param generations: for how many generations to train - aka batches
    :param edit_distance: the step by which to update - aka learning rate
    :param population: population at each step - aka how many times to try to find a better value
    :param net_class: type of network to try to train
    :param explicit_net: if not None, ANN to load and train.
    :return: trained network and training log
    """

    if explicit_net is None:
        active_net = net_class()
        active_net.main.cuda()

    else:
        active_net = explicit_net
        active_net.main.cuda()

    # uncomment:
    # active_net.main.train()

    perf_log = []

    print(active_net.model_id)

    for generation in range(generations):
        print('gen', generation, end=' - ')
        active_net, seed, loss, acc = search_best_seed(active_net, diameter=edit_distance,
                                                       tries=population, enact_update=True)
        perf_log.append([seed, loss, acc])

    v_loss, v_acc = test(active_net)

    # uncomment:
    # active_net.main.eval()

    print('validating the active net performance: %.4f, %.2f' % (v_loss, v_acc))

    return active_net, perf_log


def test_with_varying_dropout(net, drop_image=0., drop_out=0.,
                              explicit_dataloader=test_ds_loader):

    """
    Evaluates how well an ANN performs when its weight is dropped out or part of image pixels
    are dropped out

    :param net: Network to evaluate
    :param drop_image: percentage of pixels to drop out in the image
    :param drop_out: percentage of weights to drop out in the network
    :param explicit_dataloader: the dataloader supplying the data for evaluation
    :return:
    """

    buff_dropout, buff_drop_image = (active_environment.dropout,
                                     active_environment.source_image_dropout)
    active_environment.dropout = drop_out
    active_environment.source_image_dropout = drop_image

    test_net = type(net)()

    state_dict = deepcopy(net.main.state_dict())
    test_net.main.load_state_dict(state_dict)
    test_net.main.cuda()

    active_environment.dropout = buff_dropout
    active_environment.source_image_dropout = buff_drop_image

    val_loss, val_acc = test(test_net,
                             # eval_mode=False,
                             explicit_dataloader=explicit_dataloader)

    print('test DO/DI: %.2f/%.2f; loss: %.4f acc: %.2f' % (drop_out, drop_image,
                                                      val_loss, val_acc))

    return val_loss, val_acc


def flatness_sweep(active_net, max_diameter=.5, sweep_steps=10, sweep_restarts=5,
                   save_path=active_savepath,
                   explicit_dataloader=test_ds_loader):
    """
    Evaluates how flat is the minima in which the ANN s currently sitting

    :param active_net: ANN to evaluate
    :param max_diameter: maximum diameter in which to evaluate
    :param sweep_steps: how many points between 0 and max diameter to sample
    :param sweep_restarts: how many times to repeat a sweep for each point
    :param save_path: where to save the results
    :param explicit_dataloader: which data to evaluate on
    :return: combined matrix of accuracy and losses, average accuracy alone, average loss alone
    """
    combined_matrix = []
    accuracy = []
    losses = []

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Sweeps_map")

    for _ in range(sweep_restarts):
        sweep = quake(active_net,
                      diameter=[0.001, max_diameter],
                      shakes=sweep_steps,
                      explicit_dataloader=explicit_dataloader)
        sweep = np.array(sweep)
        combined_matrix.append(sweep[:, 1])
        accuracy.append(sweep[:, 1])
        combined_matrix.append(sweep[:, 2])
        losses.append(sweep[:, 2])
        diam_axis = sweep[:, 0]
        np.expand_dims(diam_axis, axis=1)

        ax1.plot(sweep[:, 0], sweep[:, 1])
        ax2.plot(sweep[:, 0], sweep[:, 2])

    ax2.set_yscale("log")

    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('diameter (stds)')

    full_path = save_path.conjugate(active_net, save_path.sweeps_map_suffix)
    plt.savefig(full_path)
    plt.clf()

    accuracy = np.array(accuracy)
    losses = np.array(losses)

    av_acc = np.average(accuracy, axis=0)
    err_acc = scst.sem(accuracy, axis=0)
    av_losses = np.average(losses, axis=0)
    err_losses = scst.sem(losses, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Sweeps_summary")

    ax1.errorbar(diam_axis, av_acc, err_acc)
    ax2.errorbar(diam_axis, av_losses, err_losses)

    ax2.set_yscale("log")

    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('diameter (stds)')

    full_path = save_path.conjugate(active_net, save_path.sweeps_summary_suffix)
    plt.savefig(full_path)
    plt.clf()

    combined_matrix = np.array(combined_matrix).T
    diam_axis = np.expand_dims(diam_axis, axis=1)

    # print(combined_matrix)
    # print(diam_axis)
    # print(combined_matrix.shape, diam_axis.shape)

    combined_matrix = np.hstack((diam_axis, combined_matrix))

    if not save_path is None:
        full_path = save_path.conjugate(active_net, save_path.flat_sweep_tsv_suffix)
        with open(full_path, 'w') as outfile:
            writer = csv_writer(outfile, delimiter='\t')
            writer.writerows(combined_matrix.tolist())

    return combined_matrix, av_acc, av_losses


if __name__ == "__main__":
    # training_trace_list = pickle.load(open('train_trace.dmp', 'br'))
    # save_and_render(training_trace_list)

    _active_net = CompressedNet
    _param_net = _active_net()

    count_parameters(_param_net)

    # print(_param_net.main.state_dict())

    # train_loop(active_environment.epochs, net_class=_active_net)

    ##################################################################
    #
    # Uncomment to run experiments with GO-EA
    #
    ##################################################################

    # Given the lenght of the execution, it is possible to perform savepoints and resume from
    # them. For that, uncomment the block below and manually provide the previous stop point save

    # previous_stop_point = 'TO PROVIDE : /run/<timestamp>/CompressedNet_<NetID>.dmp'
    #

    # _param_net.resurrect(previous_stop_point)
    # print('saved net was resurrected with id', _param_net.model_id)
    #
    # trained_net, perf_log = go_ea_train_loop(generations=800,
    #                                          edit_distance=0.01,
    #                                          population=20,
    #                                          net_class=_active_net,
    #                                          explicit_net=_param_net)
    #
    # full_path = active_savepath.conjugate(trained_net, active_savepath.go_ea_log_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerows(perf_log)
    #
    # trained_net.archive()


    ##################################################################
    #
    # Uncomment to run experiments with SGD for different param values
    #
    ##################################################################

    # trained_net_1, trace_1 = train_loop(active_environment.epochs, net_class=_active_net)
    #
    # comb_mat, av_acc, av_loss = flatness_sweep(trained_net_1, max_diameter=1.5, sweep_steps=20,
    #                                            sweep_restarts=10)

    # trained_net_2, trace_2 = train_loop(active_environment.epochs, net_class=_active_net)
    #
    # comb_mat, av_acc, av_loss = flatness_sweep(trained_net_2, max_diameter=1.5, sweep_steps=20,
    #                                            sweep_restarts=10)
    #
    # trained_net_2, trace_3 = train_loop(active_environment.epochs, net_class=_active_net)
    #
    # comb_mat, av_acc, av_loss = flatness_sweep(trained_net_2, max_diameter=1.5, sweep_steps=20,
    #                                            sweep_restarts=10)

    # print(tabulate(comb_mat.tolist()))

    # save_and_render([trace_1, trace_2, trace_3])

    ######################################################################
    #
    # Uncomment to run experiments for coding redundancy vs generalization
    #
    ######################################################################

    # annotation 2:

    # for _ in range(3):
    #
    #     trained_net_1, trace_1 = train_loop(active_environment.epochs,
    #                                         net_class=_active_net,
    #                                         explicit_train_loader=oblivious_train_ds_loader,
    #                                         explicit_test_loader=oblivious_test_ds_loader)
    #
    #     comb_mat, av_acc, av_loss = flatness_sweep(trained_net_1, max_diameter=1.5, sweep_steps=20,
    #                                                sweep_restarts=10,
    #                                                explicit_dataloader=oblivious_test_ds_loader)
    #
    #     gen_vs_rob_d = {}
    #
    #     gen_vs_rob_d['generalization'] = test_with_varying_dropout(trained_net_1, .0, .0,
    #                                                                test_ds_loader)
    #
    #     gen_vs_rob_d['robustness_10'] = test_with_varying_dropout(trained_net_1, .1, .1,
    #                                                      oblivious_test_ds_loader)
    #     gen_vs_rob_d['robustness_25'] = test_with_varying_dropout(trained_net_1, .1, .25,
    #                                                     oblivious_test_ds_loader)
    #     gen_vs_rob_d['robustness_50'] = test_with_varying_dropout(trained_net_1, .1, .5,
    #                                                               oblivious_test_ds_loader)
    #
    #     save_path = active_savepath.conjugate(trained_net_1, active_savepath.gener_vs_redun_suffix)
    #
    #     with open(save_path, 'w') as outfile:
    #         yaml.dump(gen_vs_rob_d, outfile, default_flow_style=False)


    ##################################################################
    #
    # Uncomment to run experiments for fine-tuning effect on minima
    #
    ##################################################################

    # # annotation 3:
    #
    # for _ in range(3):
    #
    #     trained_net_1, trace_1 = train_loop(active_environment.epochs,
    #                                         net_class=_active_net,
    #                                         explicit_train_loader=oblivious_train_ds_loader,
    #                                         explicit_test_loader=oblivious_test_ds_loader)
    #
    #     comb_mat, av_acc, av_loss = flatness_sweep(trained_net_1, max_diameter=1.5, sweep_steps=20,
    #                                                sweep_restarts=5,
    #                                                explicit_dataloader=oblivious_test_ds_loader)
    #
    #     trained_net_1.fine_tuned_id()
    #
    #     tuned_net_1_1, trace_1_1 = train_loop(active_environment.fine_tune_epochs,
    #                                       net_class=_active_net,
    #                                       explicit_net=trained_net_1)
    #
    #     comb_mat, av_acc, av_loss = flatness_sweep(tuned_net_1_1, max_diameter=1.5, sweep_steps=20,
    #                                                sweep_restarts=5)
    #
    #     tuned_net_1_1.fine_tuned_id()
    #
    #     tuned_net_1_2, trace_1_1 = train_loop(active_environment.fine_tune_epochs,
    #                                       net_class=_active_net,
    #                                       explicit_net=tuned_net_1_1)
    #
    #     comb_mat, av_acc, av_loss = flatness_sweep(tuned_net_1_2, max_diameter=1.5, sweep_steps=20,
    #                                                sweep_restarts=5)
    #
    #     tuned_net_1_2.fine_tuned_id()
    #
    #     tuned_net_1_3, trace_1_1 = train_loop(active_environment.fine_tune_epochs,
    #                                       net_class=_active_net,
    #                                       explicit_net=tuned_net_1_2)
    #
    #     comb_mat, av_acc, av_loss = flatness_sweep(tuned_net_1_3, max_diameter=1.5, sweep_steps=20,
    #                                                sweep_restarts=5)

    ##################################################################
    #
    # GO-EA parameteric sweeps
    #
    ##################################################################

    # base = 'TO PROVIDE : /run/<timestamp>/CompressedNet_<NetID>.dmp'

    # # change the learning rate: x8, /8
    #
    # _interim_net_1 = _active_net()
    # _interim_net_1.resurrect(base)
    #
    # _interim_net_1.fine_tuned_id()
    #
    # trained_net, perf_log = go_ea_train_loop(generations=30,
    #                                          edit_distance=0.08,
    #                                          population=20,
    #                                          net_class=_active_net,
    #                                          explicit_net=_interim_net_1)
    #
    # full_path = active_savepath.conjugate(trained_net, active_savepath.go_ea_log_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerows(perf_log)
    #
    # trained_net.archive()
    #
    #
    # _interim_net_2 = _active_net()
    # _interim_net_2.resurrect(base)
    #
    # _interim_net_2.fine_tuned_id()
    #
    # trained_net, perf_log = go_ea_train_loop(generations=30,
    #                                          edit_distance=0.00125,
    #                                          population=20,
    #                                          net_class=_active_net,
    #                                          explicit_net=_interim_net_2)
    #
    # full_path = active_savepath.conjugate(trained_net, active_savepath.go_ea_log_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerows(perf_log)
    #
    # trained_net.archive()
    #
    # # change the population size: 3; 160
    #
    # _interim_net_3 = _active_net()
    # _interim_net_3.resurrect(base)
    #
    # _interim_net_3.fine_tuned_id()
    #
    # trained_net, perf_log = go_ea_train_loop(generations=30,
    #                                          edit_distance=0.01,
    #                                          population=40,
    #                                          net_class=_active_net,
    #                                          explicit_net=_interim_net_3)
    #
    # full_path = active_savepath.conjugate(trained_net, active_savepath.go_ea_log_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerows(perf_log)
    #
    # trained_net.archive()
    #
    #
    # _interim_net_4 = _active_net()
    # _interim_net_4.resurrect(base)
    #
    # _interim_net_4.fine_tuned_id()
    #
    # trained_net, perf_log = go_ea_train_loop(generations=30,
    #                                          edit_distance=0.01,
    #                                          population=4,
    #                                          net_class=_active_net,
    #                                          explicit_net=_interim_net_4)
    #
    # full_path = active_savepath.conjugate(trained_net, active_savepath.go_ea_log_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerows(perf_log)
    #
    # trained_net.archive()


    ##################################################################
    #
    # random update vector behavior
    #
    ##################################################################

    # make sure the model is trained without drop-out and #evolve tags.

    # _interim_net_1 = _active_net()
    #
    #
    # loss_list, acc_list, linearized_tensor_list = perform_circular_sweep(_interim_net_1,
    #                                                                      diameter=0.01,
    #                                                                      tries=100)
    #
    # full_path = active_savepath.conjugate(_interim_net_1, active_savepath.circular_sweep_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerow(loss_list.tolist())
    #     writer.writerow(acc_list.tolist())
    #     writer.writerows(linearized_tensor_list.tolist())
    #
    # _interim_net_1.archive()
    #
    #
    # _interim_net_1.fine_tuned_id()
    #
    # _interim_net_1, trace_1 = train_loop(active_environment.epochs,
    #                                     net_class=_active_net,
    #                                     explicit_net=_interim_net_1,
    #                                     explicit_train_loader=oblivious_train_ds_loader,
    #                                     explicit_test_loader=oblivious_test_ds_loader)
    #
    #
    #
    # loss_list, acc_list, linearized_tensor_list = perform_circular_sweep(_interim_net_1,
    #                                                                      diameter=0.01,
    #                                                                      tries=100)
    #
    # full_path = active_savepath.conjugate(_interim_net_1, active_savepath.circular_sweep_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerow(loss_list.tolist())
    #     writer.writerow(acc_list.tolist())
    #     writer.writerows(linearized_tensor_list.tolist())
    #
    #
    # _interim_net_1.fine_tuned_id()
    #
    # _interim_net_1, trace_1 = train_loop(active_environment.epochs,
    #                                     net_class=_active_net,
    #                                     explicit_net=_interim_net_1)
    #
    # loss_list, acc_list, linearized_tensor_list = perform_circular_sweep(_interim_net_1,
    #                                                                      diameter=0.01,
    #                                                                      tries=100)
    #
    # full_path = active_savepath.conjugate(_interim_net_1, active_savepath.circular_sweep_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerow(loss_list.tolist())
    #     writer.writerow(acc_list.tolist())
    #     writer.writerows(linearized_tensor_list.tolist())


    ##################################################################
    #
    # batch size effect on angle divergence
    #
    ##################################################################

    # make sure the model is trained without drop-out and #evolve tags.

    # _interim_net_1 = _active_net()
    #
    #
    # full_path = active_savepath.conjugate(_interim_net_1, active_savepath.sgd_angle)
    #
    #
    # _interim_net_1.fine_tuned_id()
    #
    # _interim_net_1, trace_1 = train_loop(active_environment.epochs,
    #                                     net_class=_active_net,
    #                                     explicit_net=_interim_net_1,
    #                                     explicit_train_loader=oblivious_train_ds_loader,
    #                                     explicit_test_loader=oblivious_test_ds_loader)
    #
    #
    # _norm_list , _angle_matrix = sweep_batch_updates(explicit_net=_interim_net_1)
    #
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerow(_norm_list.tolist())
    #     writer.writerows(_angle_matrix.tolist())

    #
    # full_path = active_savepath.conjugate(_interim_net_1, active_savepath.circular_sweep_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerow(loss_list.tolist())
    #     writer.writerow(acc_list.tolist())
    #     writer.writerows(linearized_tensor_list.tolist())
    #
    #
    # _interim_net_1.fine_tuned_id()
    #
    # _interim_net_1, trace_1 = train_loop(active_environment.epochs,
    #                                     net_class=_active_net,
    #                                     explicit_net=_interim_net_1)
    #
    # loss_list, acc_list, linearized_tensor_list = perform_circular_sweep(_interim_net_1,
    #                                                                      diameter=0.01,
    #                                                                      tries=100)
    #
    # full_path = active_savepath.conjugate(_interim_net_1, active_savepath.circular_sweep_suffix)
    # with open(full_path, 'w') as outfile:
    #     writer = csv_writer(outfile, delimiter='\t')
    #     writer.writerow(loss_list.tolist())
    #     writer.writerow(acc_list.tolist())
    #     writer.writerows(linearized_tensor_list.tolist())


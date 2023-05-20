import os
from pathlib import Path
from matplotlib import pyplot
import yaml
from pprint import pprint
from tabulate import tabulate

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import stats as scst
from scipy.stats import gaussian_kde, ks_2samp
import scipy.special as sc

from csv import reader as csv_reader
from collections import defaultdict, OrderedDict


def parse_flatness_experiments(runs_save_directory):
    """
    Function that parses flatness evaluation experiments contained in a directory

    :param runs_save_directory: directory containing the experiments
    :return:
    """
    experiments_list = []

    for name in os.listdir(runs_save_directory):
        # print(name)
        name_save_dir = runs_save_directory / name

        if os.path.isdir(name_save_dir):
            experiment = {}

            for _fl in os.listdir(name_save_dir):
                # print('\t', _fl)
                _fl_path = name_save_dir / _fl

                if _fl == 'training_params.yaml':
                    experiment['settings'] = yaml.load(open(_fl_path, 'r'))
                    experiment['settings']['location'] = name

                if _fl_path.suffix == '.dmp':
                    replicate = {}
                    replicate['trained_model'] = _fl_path
                    stem = _fl_path.stem
                    # print('training run %s detected, checking for traces' % stem)

                    for secondary_suffix in ['train_trace',
                                             'flat_sweep_combined',
                                             'circular_sweep']:
                        full_suffix = stem + '_' + secondary_suffix + '.tsv'
                        full_suffixed_path = name_save_dir / full_suffix
                        if full_suffixed_path.is_file():
                            replicate[secondary_suffix] = full_suffixed_path

                    for secondary_suffix in ['generalization_vs_redundancy']:
                        full_suffix = stem + '_' + secondary_suffix + '.yaml'
                        full_suffixed_path = name_save_dir / full_suffix
                        if full_suffixed_path.is_file():
                            replicate[secondary_suffix] = yaml.load(open(full_suffixed_path, 'r'))

                    if len(replicate.keys()) > 2:
                        experiment[stem] = replicate

            if len(experiment.keys()) > 3:  # setting + at least 3 replicates
                experiments_list.append(experiment)

    # pprint(experiments_list)

    return experiments_list


def parse_go_ea_experiments(runs_save_dir,
                            go_ea_net,
                            go_ea_sweep_dir,
                            go_ea_sweep_variations,
                            go_ea_main_trunk_dir,
                            go_ea_main_track_dirs):
    """
    Function that parses GO-EA experiments contained in a directory

    :param runs_save_dir:
    :param go_ea_net:
    :param go_ea_sweep_dir:
    :param go_ea_sweep_variations:
    :param go_ea_main_trunk_dir:
    :param go_ea_main_track_dirs:
    :return:
    """

    acc_loss_column = []

    for directory_stem in go_ea_main_track_dirs:
        completed_suffix = go_ea_net + '_GO_EA_log.tsv'
        full_path = runs_save_dir / directory_stem / completed_suffix
        acc_loss_table = np.genfromtxt(full_path,
                                       delimiter='\t',
                                       converters={0: lambda s: float(s or -1)}
                                       )
        acc_loss_column.append(acc_loss_table)

    acc_loss_column = np.vstack(tuple(acc_loss_column))

    loss_column = acc_loss_column[:, 1]
    acc_column = acc_loss_column[:, 2]

    fig, ax1 = plt.subplots()
    fig.suptitle('GO-EA training trace')

    ax2 = ax1.twinx()

    contrast_c = cm.get_cmap(active_cmap)(0.8)

    ax2.plot(range(len(loss_column)), loss_column, c=contrast_c, label='loss')
    ax1.plot(range(len(acc_column)), acc_column, 'k', label='accuracy')
    ax1.plot([], c=contrast_c, label='loss')

    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax1.legend(loc='lower center', ncol=2)
    # fig.legend([line1, line2], ['loss', 'acc'], loc=(0.5, 0), ncol=2 )
    ax1.set_xlabel('generation')

    # fig.set_size_inches(24, 13.5, forward=True)  # 1080p
    fig.set_dpi(160)

    plt.show()


    experimental_dict = OrderedDict()

    completed_suffix = go_ea_net + '_GO_EA_log.tsv'
    full_path = runs_save_dir / go_ea_main_trunk_dir / completed_suffix
    experimental_dict['base'] = np.genfromtxt(full_path,
                                                   delimiter='\t',
                                                   converters={0: lambda s: float(s or -1)})


    for exp_name, exp_suffix in go_ea_sweep_variations.items():
        completed_suffix = go_ea_net + exp_suffix + '_GO_EA_log.tsv'
        full_path = runs_save_dir / go_ea_sweep_dir / completed_suffix
        experimental_dict[exp_name] = np.genfromtxt(full_path,
                                                       delimiter='\t',
                                                       converters={0: lambda s: float(s or -1)})


    comparison_colors = generate_comparison_colors(len(experimental_dict.keys())-1)
    comparison_colors = ['k'] + comparison_colors

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('GO-EA variations comparison')

    for i, (exp_name, exp_trace) in enumerate(experimental_dict.items()):

        loss_column = exp_trace[:, 1]
        acc_column = exp_trace[:, 2]

        ax1.plot(acc_column, color=comparison_colors[i], label=exp_name)
        ax2.plot(loss_column, color=comparison_colors[i], label=exp_name)


    # ax2.set_yscale('log')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax2.legend(ncol=2)
    ax2.set_xlabel('generations')

    # fig.set_size_inches(24, 13.5, forward=True)  # 1080p
    fig.set_dpi(160)

    plt.show()


def normalize_experiments(experiments_list):
    """
    Performs a normalization of experiments

    :param experiments_list:
    :return:
    """

    for experiment in experiments_list:
        if 'linear_width' not in experiment['settings'].keys():
            experiment['settings']['linear_width'] = 24
        if 'source_image_dropout' not in experiment['settings'].keys():
            experiment['settings']['source_image_dropout'] = experiment['settings']['dropout']


def inspect_experiments(experiments_list):
    """
    Looks at experiments contained in a folder to try to guess the type of experiments that
    were performed

    :param experiments_list:
    :return:
    """


    possible_parameters = [
            'classifier_latent_maps',
            'linear_width',
            'dropout',
            'source_image_dropout',
            'epochs',
            # 'image_size',
            'learning_rate',
            'momentum',
            'test_batch_size',
            'train_batch_size',
            'annotation',
            'location',
            'fine_tune_epochs',
    ]

    param_name_2_idx = {name: i for i, name in enumerate(possible_parameters)}
    idx_2_param_name = {i: name for i, name in enumerate(possible_parameters)}

    shape = len(possible_parameters)+1

    experiment_accumulator = []

    for i, experiment in enumerate(experiments_list):
        template = [-1]*shape
        for key, value in experiment['settings'].items():
            if key in possible_parameters:
                template[param_name_2_idx[key]+1] = value
        template[0] = i
        experiment_accumulator.append(template)

    print(tabulate(experiment_accumulator, headers=possible_parameters))

    return experiment_accumulator


def tsv_to_numpy(tsv_file_path):
    """
    Parses a .tsv file into a numpy file

    :param tsv_file_path:
    :return:
    """

    lines = []
    with open(tsv_file_path, 'r') as infile:
        reader = csv_reader(infile, delimiter='\t')
        for row in reader:
            lines.append(row)

    block = np.array(lines)

    diameters = np.array(block[:, 0]).astype(np.float)

    pair_indices = [True if i % 2 == 0 else False for i in range(0, block.shape[1])]
    pair_indices[0] = False  # it's diameter

    impair_indices = [not b_o for b_o in pair_indices]
    impair_indices[0] = False  # it's still diameter

    accuracy = np.array(block[:, impair_indices]).astype(np.float)
    losses = np.array(block[:, pair_indices]).astype(np.float)


    return diameters, accuracy, losses


def compare_training_traces(experiments_list,
                            base_index,
                            comparison_indexes):
    pass


def compare_flatness(experiments_list,
                     base_index,
                     comparison_indexes,
                     forced_base_legend=None,
                     forced_legends=None,
                     explicit_title="Sweeps_comparison",
                     fine_tune_steps=0,
                     fine_tune_sweep=False):
    """
    Draws figures that compare flatness experiments

    :param experiments_list: list of experiments
    :param base_index: index of experiment that's considered as a reference
    :param comparison_indexes: indexes of experiments that are considered as variations
    :param forced_base_legend: the legend to override reference experiment description
    :param forced_legends: legends list to override experiments description
    :param explicit_title: title to give to a plot
    :param fine_tune_steps: if fine-tuning, number of steps
    :param fine_tune_sweep: if fine-tuning sweeps needs to be annotated
    :return: None
    """


    comparison_colors = generate_comparison_colors(len(comparison_indexes))


    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(explicit_title)


    current_legend = 'ref: %d' % base_index
    if forced_base_legend is not None:
        current_legend = forced_base_legend

    for replicate_name, replicate in experiments_list[base_index].items():

        if replicate_name == 'settings':
            continue

        if fine_tune_steps > -1:
            replicate_fine_tunes = replicate_name.count("_ft_")
            if replicate_fine_tunes != fine_tune_steps:
                continue

        diameters, accuracy, losses = tsv_to_numpy(replicate['flat_sweep_combined'])

        av_acc = np.average(accuracy, axis=1)
        err_acc = scst.sem(accuracy, axis=1)
        av_losses = np.average(losses, axis=1)
        err_losses = scst.sem(losses, axis=1)

        ax1.errorbar(diameters, av_acc, err_acc,
                     color='k', label=current_legend)
        ax2.errorbar(diameters, av_losses, err_losses,
                     color='k', label=current_legend)

        if current_legend != '_nolegend_':
            current_legend = '_nolegend_'


    for i, comparison_index in enumerate(comparison_indexes):

        if fine_tune_sweep:
            fine_tune_steps += 1

        current_legend = 'exp %d' % comparison_index
        if forced_legends is not None:
            current_legend = forced_legends[i]

        for replicate_name, replicate in experiments_list[comparison_index].items():

            if replicate_name == 'settings':
                continue

            if fine_tune_steps > -1:
                replicate_fine_tunes = replicate_name.count("_ft_")
                if replicate_fine_tunes != fine_tune_steps:
                    continue

            diameters, accuracy, losses = tsv_to_numpy(replicate['flat_sweep_combined'])

            av_acc = np.average(accuracy, axis=1)
            err_acc = scst.sem(accuracy, axis=1)
            av_losses = np.average(losses, axis=1)
            err_losses = scst.sem(losses, axis=1)

            ax1.errorbar(diameters, av_acc, err_acc,
                         color=comparison_colors[i], label=current_legend)
            ax2.errorbar(diameters, av_losses, err_losses,
                         color=comparison_colors[i], label=current_legend)

            if current_legend != '_nolegend_':
                current_legend = '_nolegend_'

    ax2.set_yscale('log')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax1.legend()
    ax2.set_xlabel('diameter (stds)')

    # fig.set_size_inches(24, 13.5, forward=True)  # 1080p
    fig.set_dpi(160)

    plt.show()

    # full_path = save_path.conjugate(active_net, save_path.sweeps_summary_suffix)
    # plt.savefig(full_path)
    # plt.clf()


def compare_robustness(experiments_list,
                     base_index,
                     comparison_indexes,
                     forced_base_legend=None,
                     forced_legends=None,
                     explicit_title="Robustness comparison"):
    """
    Draws the boxplot that compares robustness of different ANN architectures upon fine-tune

    :param experiments_list: list of experiments as indexes in folder list
    :param base_index: index of the reference experiment
    :param comparison_indexes: indexes of experiemnts to which the reference is compared to
    :param forced_base_legend: the legend override for the reference index
    :param forced_legends: legend overrides for the experiments indexes
    :param explicit_title: title to give to the plot
    :return: none
    """

    def set_box_color(bp, color):
        """
        Sets the color of the boxplot bp to a given color

        :param bp: boxplot object
        :param color: color to set it to
        :return:
        """
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    comparison_colors = generate_comparison_colors(len(comparison_indexes))
    comparison_colors = ['k'] + comparison_colors

    ticks = ['gen', 'rob 0.1', 'rob 0.25', 'rob 0.5']

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(explicit_title)

    for i, exp_index in enumerate([base_index] + comparison_indexes):

        acc_aggregator = [[], [], [], []]
        loss_aggregator = [[], [], [], []]

        for replicate_name, replicate in experiments_list[exp_index].items():
            if replicate_name == 'settings':
                continue

            acc_aggregator[0].append(replicate['generalization_vs_redundancy']['generalization'][1])
            acc_aggregator[1].append(replicate['generalization_vs_redundancy']['robustness_10'][1])
            acc_aggregator[2].append(replicate['generalization_vs_redundancy']['robustness_25'][1])
            acc_aggregator[3].append(replicate['generalization_vs_redundancy']['robustness_50'][1])

            loss_aggregator[0].append(replicate['generalization_vs_redundancy'][
                                          'generalization'][0])
            loss_aggregator[1].append(replicate['generalization_vs_redundancy']['robustness_10'][0])
            loss_aggregator[2].append(replicate['generalization_vs_redundancy']['robustness_25'][0])
            loss_aggregator[3].append(replicate['generalization_vs_redundancy']['robustness_50'][0])

        # print(acc_aggregator)
        # print(np.array(range(len(acc_aggregator)))*3.0-0.8+i*0.8)

        bp1 = ax1.boxplot(acc_aggregator,
                          positions=np.array(range(len(acc_aggregator)))*3.0-0.8+i*0.8,
                          sym='',
                          widths=0.6)

        bp2 = ax2.boxplot(loss_aggregator,
                          positions=np.array(range(len(loss_aggregator)))*3.0-0.8+i*0.8,
                          sym='',
                          widths=0.6)

        set_box_color(bp1, comparison_colors[i])
        set_box_color(bp2, comparison_colors[i])

        if i == 0:
            current_legend = 'ref: %d' % base_index
            if forced_base_legend is not None:
                current_legend = forced_base_legend
        else:
            current_legend = 'exp %d' % exp_index
            if forced_legends is not None:
                current_legend = forced_legends[i-1]

        ax1.plot([], c=comparison_colors[i], label=current_legend)


    ax1.set_xticks(range(0, len(ticks) * 3, 3))
    ax1.set_xticklabels(ticks)

    ax2.set_xticks(range(0, len(ticks) * 3, 3))
    ax2.set_xticklabels(ticks)

    ax2.set_yscale('log')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax1.legend()
    fig.set_dpi(160)

    plt.show()


def compare_fine_tune_speed(experiments_list,
                            base_index,
                            comparison_indexes,
                            forced_base_legend=None,
                            forced_legends=None,
                            explicit_title="Fine Tune speed comparison",
                            assumed_fine_tunes=3):
    """
    Function to draw the comparison of the fine-tuning speeds

    :param experiments_list: list of experiments to include in the analysis
    :param base_index: index of the reference run
    :param comparison_indexes: indexes of experiments to compare to the reference
    :param forced_base_legend: legend to override the default one for the reference run
    :param forced_legends: legends to override the default ones for the experiments runs
    :param explicit_title: title to give to the plot
    :param assumed_fine_tunes: how many epoch of fine-tune was performed at the end of experiments
    :return: None
    """

    comparison_colors = generate_comparison_colors(len(comparison_indexes))

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(explicit_title)


    accuracy_dict = defaultdict(lambda: [[]]*(assumed_fine_tunes + 1))
    loss_dict = defaultdict(lambda: [[]]*(assumed_fine_tunes + 1))

    for replicate_name, replicate in experiments_list[base_index].items():

        if replicate_name == 'settings':
            continue

        replicate_fine_tunes = replicate_name.count("_ft_")
        replicate_root = replicate_name.split("_ft_")[0]

        diameters, accuracy, losses = tsv_to_numpy(replicate['flat_sweep_combined'])

        av_acc = np.average(accuracy, axis=1)[0]
        err_acc = scst.sem(accuracy, axis=1)[0]
        av_losses = np.average(losses, axis=1)[0]
        err_losses = scst.sem(losses, axis=1)[0]

        accuracy_dict[replicate_root][replicate_fine_tunes] = [av_acc, err_acc]
        loss_dict[replicate_root][replicate_fine_tunes] = [av_losses, err_losses]

    current_legend = 'ref: %d' % base_index
    if forced_base_legend is not None:
        current_legend = forced_base_legend


    for replicate_root in accuracy_dict.keys():

        diameters = range(assumed_fine_tunes+1)

        av_acc = [av for (av, err) in accuracy_dict[replicate_root]]
        err_acc = [err for (av, err) in accuracy_dict[replicate_root]]

        av_losses = [av for av, err in loss_dict[replicate_root]]
        err_losses = [err for av, err in loss_dict[replicate_root]]

        ax1.errorbar(diameters, av_acc, err_acc,
                     color='k', label=current_legend)
        ax2.errorbar(diameters, av_losses, err_losses,
                     color='k', label=current_legend)

        if current_legend != '_nolegend_':
            current_legend = '_nolegend_'


    for i, comparison_index in enumerate(comparison_indexes):

        accuracy_dict = defaultdict(lambda: [[]]*(assumed_fine_tunes + 1))
        loss_dict = defaultdict(lambda: [[]]*(assumed_fine_tunes + 1))

        current_legend = 'exp %d' % comparison_index
        if forced_legends is not None:
            current_legend = forced_legends[i]

        for replicate_name, replicate in experiments_list[comparison_index].items():

            if replicate_name == 'settings':
                continue

            replicate_fine_tunes = replicate_name.count("_ft_")
            replicate_root = replicate_name.split("_ft_")[0]

            diameters, accuracy, losses = tsv_to_numpy(replicate['flat_sweep_combined'])

            av_acc = np.average(accuracy, axis=1)[0]
            err_acc = scst.sem(accuracy, axis=1)[0]
            av_losses = np.average(losses, axis=1)[0]
            err_losses = scst.sem(losses, axis=1)[0]

            accuracy_dict[replicate_root][replicate_fine_tunes] = [av_acc, err_acc]
            loss_dict[replicate_root][replicate_fine_tunes] = [av_losses, err_losses]


        current_legend = 'exp %d' % comparison_index
        if forced_legends is not None:
            current_legend = forced_legends[i]

        for replicate_root in accuracy_dict.keys():

            diameters = range(assumed_fine_tunes+1)

            av_acc = [av for av, err in accuracy_dict[replicate_root]]
            err_acc = [err for av, err in accuracy_dict[replicate_root]]

            av_losses = [av for av, err in loss_dict[replicate_root]]
            err_losses = [err for av, err in loss_dict[replicate_root]]

            ax1.errorbar(diameters, av_acc, err_acc,
                         color=comparison_colors[i], label=current_legend)
            ax2.errorbar(diameters, av_losses, err_losses,
                         color=comparison_colors[i], label=current_legend)

            if current_legend != '_nolegend_':
                current_legend = '_nolegend_'

    ticks = range(0, assumed_fine_tunes+1)

    ax2.set_xticks(range(0, len(ticks)))
    ax2.set_xticklabels(ticks)

    ax2.set_yscale('log')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax1.legend()
    ax2.set_xlabel('epochs')

    # fig.set_size_inches(24, 13.5, forward=True)  # 1080p
    fig.set_dpi(160)

    plt.show()


def parse_circular_sweep_dirs(runs_save_directory,
                              sweep_directory,
                              tune_status_titles=None):
    """
    This function parses circular sweeps experiments

    :param runs_save_directory: directory of runs where circular sweeps were performed
    :param sweep_directory: directory of circular sweeps within the directory
    :param tune_status_titles: classification by fine-tune status
    :return:
    """

    if tune_status_titles is None:
        tune_status_titles = ['TS 1', 'TS 2', 'TS 3']

    comparison_colors = generate_comparison_colors(len(tune_status_titles)+1)
    # comparison_colors = ['k'] + comparison_colors

    tune_status_dict = {}

    acc_max, acc_min = (0, 0)
    loss_max, loss_min = (0, 0)

    for fle in os.listdir(runs_save_directory / sweep_directory):
        if not '_circular_sweep.tsv' in fle:
            continue
        fine_tune_status = fle.count('_ft_')
        aspiration_path = runs_save_directory / sweep_directory / fle
        loss_acc_g_angle = np.genfromtxt(aspiration_path,
                                         delimiter='\t',
                                         converters={0: lambda s: float(s or np.nan)},
                                         encoding='UTF-8')
        loss_g = loss_acc_g_angle[0, :]
        acc_g = loss_acc_g_angle[1, :]
        angles = loss_acc_g_angle[2:, :]

        acc_max = np.max([acc_max, np.max(acc_g)])
        acc_min = np.min([acc_min, np.min(acc_g)])

        loss_max = np.max([loss_max, np.max(loss_g)])
        loss_min = np.min([loss_min, np.min(loss_g)])

        angles = np.abs(np.triu(angles, k=1))

        selec = loss_g < 0

        selec_pos = np.where(selec)

        # print(selec_pos)

        better_angles = []
        for pos_l in selec_pos[0]:
            for pos_c in selec_pos[0]:
                print('\t', pos_l, pos_c)
                if angles[pos_l, pos_c] != 0:
                    better_angles.append(angles[pos_l, pos_c])
            # print('\t', pos)
            # vert_angles = angles[pos, :]
            # select_vert_angles = vert_angles[better_angles]
            # select_vert_angles = select_vert_angles.tolist()
            # better_angles += select_vert_angles
            # better_angles += angles[pos, better_angles].tolist()

        better_angles = np.array(better_angles)
        better_angles = better_angles[better_angles != 0]

        tune_status_dict[fine_tune_status] = (loss_g, acc_g, angles, better_angles)

        # print(better_angles)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Possible improvements through a GO-EA step')

    for i, (tune_status, stats) in enumerate(tune_status_dict.items()):

        loss_g, acc_g, angles, better_angles = stats

        loss_d = gaussian_kde(loss_g)
        xs = np.linspace(loss_min, loss_max, 100)
        loss_d.covariance_factor = lambda: .25
        loss_d._compute_covariance()
        ax2.plot(xs, loss_d(xs),
                 color=comparison_colors[tune_status+1],
                 label=tune_status_titles[tune_status])

        acc_d = gaussian_kde(acc_g)
        xs = np.linspace(acc_min, acc_max, 100)
        acc_d.covariance_factor = lambda: .25
        acc_d._compute_covariance()
        ax1.plot(xs, acc_d(xs),
                 color=comparison_colors[tune_status+1],
                 label=tune_status_titles[tune_status])

    ax1.axvline(x=0, color=comparison_colors[0])
    ax2.axvline(x=0, color=comparison_colors[0])

    # ax2.set_yscale('log')
    ax1.legend()
    ax1.set_ylabel('distribution density')
    ax2.set_ylabel('distribution density')

    ax1.set_xlabel('accuracy %% gained (more is better)')
    ax2.set_xlabel('loss gained (less is better)')

    ax2.invert_xaxis()
    # fig.set_size_inches(24, 13.5, forward=True)  # 1080p
    fig.set_dpi(160)
    plt.tight_layout()

    plt.show()

    input('press enter')

    contrast_c = cm.get_cmap(active_cmap)(0.8)

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Angle between vectors offering improvement')

    for i, (tune_status, stats) in enumerate(tune_status_dict.items()):

        loss_g, acc_g, angles, better_angles = stats

        ang = angles[angles != 0].flatten()
        ang_d = gaussian_kde(ang)
        xs = np.linspace(0, 90, 100)
        ang_d.covariance_factor = lambda: .25
        ang_d._compute_covariance()

        if tune_status < 2:
            ang_b = better_angles.flatten()
            ang_b_d = gaussian_kde(ang_b)
            xs = np.linspace(0, 90, 100)
            ang_b_d.covariance_factor = lambda: .25
            ang_b_d._compute_covariance()

            # print(ang)
            # print(ang_b)

            ks_value = ks_2samp(ang, ang_b)

            # print(ks_value)

        if tune_status == 0:
            ax1.plot(xs, ang_d(xs), 'k', label='all vectors')
            ax1.plot(xs, ang_b_d(xs), c=contrast_c, label='vectors with improvement')
            # print(tune_status, ks_value)
            ax1.text(50,
                     # 1e-100,
                     0.5,
                     'KS p-val: %.2f' % ks_value[1])


        if tune_status == 1:
            ax2.plot(xs, ang_d(xs), 'k', label='all vectors')
            ax2.plot(xs, ang_b_d(xs), c=contrast_c, label='vectors with improvement')
            # print(tune_status, ks_value)
            ax2.text(50,
                     # 1e-100,
                     0.5,
                     'KS p-val: %.2f' % ks_value[1])

        if tune_status == 2:
            ax3.plot(xs, ang_d(xs), c='k', label='all vectors')
            # ax3.plot(xs, ang_b_d(xs), 'r', label='vectors with improvement')


    ax2.legend()

    # ax1.set_yscale("log")
    # ax2.set_yscale("log")
    # ax3.set_yscale("log")

    ax1.set_ylabel('pre train\n(d of abs)')
    ax2.set_ylabel('pre fine-tune\n(d of abs)')
    ax3.set_ylabel('post fine-tune\n(d of abs)')

    ax3.set_xlabel('angle between vectors')

    # fig.set_size_inches(24, 13.5, forward=True)  # 1080p
    fig.set_dpi(160)
    plt.tight_layout()

    plt.show()


def generate_comparison_colors(data_size=8):
    """
    Generate the colors for the experiments to compare to the reference

    :param data_size:
    :return:
    """
    if colors_locked:
        data_size = 8

    return [cm.get_cmap(active_cmap)(i) for i in np.linspace(1 / data_size, .99, data_size)]


def generate_dim_sampling_offset_sweep():
    """
    Generates the figures that show how the chance of sampling a direction within x of
    the source

    :return:
    """

    def relative_cap_size(angle, dim):

        # print(angle, dim)
        angle_sin = np.sin(np.deg2rad(angle))
        # print("\t", angle_sin)
        # print("\t", np.power(angle_sin, 2))
        val_reg = 0.5 * sc.betainc((dim - 1) / 2, 1/2, np.power(angle_sin, 2))*100
        # print("\t", val_reg)
        return val_reg

    rel_cap_vectorized = np.vectorize(relative_cap_size)

    fig, (ax1, ax2) = plt.subplots(2,)
    fig.suptitle('Chance to land within α degs from a reference vector in dim n')


    x_dims = np.logspace(1, 3, 100)  # TODO: switch to log
    # print(x_dims)
    # input("press enter")
    x_angles = np.linspace(90, 0.00, 100)

    cm = generate_comparison_colors(4)
    angles = [5, 10, 30, 60]
    dims = [3, 10, 100, 1000]

    for i, angle in enumerate(angles):
        ax1.plot(x_dims, rel_cap_vectorized(angle, x_dims), c=cm[i], label="within %d deg" % angle)

    for i, dim in enumerate(dims):
        ax2.plot(x_angles, rel_cap_vectorized(x_angles, dim), c=cm[i], label="in dim %d" % dim)

    ax1.set_ylabel("%% chance to land")
    ax1.set_xlabel("in dimension")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-9, 1)
    ax1.set_xscale("log")
    ax1.legend()

    ax2.set_ylabel("%% chance to land")
    ax2.set_xlabel("within angle (deg)")
    ax2.legend()

    plt.tight_layout()
    fig.set_dpi(160)

    plt.show()


def parse_sgd_update_tensor_dirs(runs_save_directory,
                                 sweep_directories,
                                 batch_titles=None):
    """
    This function parses and renders the sgd update divergence experiment

    :param runs_save_directory: directory of runs where circular sweeps were performed
    :param sweep_directories: directory of circular sweeps within the directory
    :param batch_titles: classification by fine-tune status
    :return:
    """

    def get_from_text_with_sampling_failover(_aspiration_path):
        """
        Because data is big and RAM is small

        :return:
        """
        sample_flag = False

        with open(_aspiration_path, 'rt') as source_fle:
            reader = csv_reader(source_fle, delimiter='\t')
            norms = reader.__next__()
            if len(norms) > limit_dims:
                sample_flag = True

        if not sample_flag:

            norm_angle = np.genfromtxt(aspiration_path,
                                   delimiter='\t',
                                   converters={0: lambda s: float(s or np.nan)},
                                   encoding='UTF-8')

            norms = norm_angle[0, :]
            angles = norm_angle[1:, :]

            return norms, angles

        with open(_aspiration_path, 'rt') as source_fle:
            reader = csv_reader(source_fle, delimiter='\t')
            norms = reader.__next__()
            ref_size = len(norms)
            random_pad = np.random.randint(ref_size, size=limit_dims)
            norms = np.array(norms)[random_pad].astype(np.float)
            random_pad = np.random.randint(ref_size, size=limit_dims).tolist()
            angles = []

            for i, line in enumerate(reader):
                if not i in random_pad:
                    continue

                re_random_pad = np.random.randint(ref_size, size=limit_dims)
                line = np.array(line)[re_random_pad].astype(np.float)
                angles.append(line)

            angles = np.array(angles)

            return norms, angles


    if batch_titles is None:
        batch_titles = ['BS %d' % i for i, _ in enumerate(sweep_directories)]

    comparison_colors = generate_comparison_colors(len(batch_titles))

    batch_sizes_sweeps = {}

    for title, sweep_directory in zip(batch_titles, sweep_directories):

        mem_fle = None
        for fle in os.listdir(runs_save_directory / sweep_directory):
            if not 'sgd_angle.tsv' in fle:
                continue
            else:
                mem_fle = fle

        aspiration_path = runs_save_directory / sweep_directory / mem_fle
        print('loading %s' % aspiration_path)


        limit_dims = 2000

        norms, angles = get_from_text_with_sampling_failover(aspiration_path)

        # print(norms.shape)
        # print(angles.shape)


        # norm_angle = np.genfromtxt(aspiration_path,
        #                            delimiter='\t',
        #                            converters={0: lambda s: float(s or np.nan)},
        #                            encoding='UTF-8')
        #
        # if norm_angle.shape[0] > limit_dims:
        #     random_pad = np.random.randint(norm_angle.shape[0]-1, size=limit_dims)
        #     norms = norm_angle[0, random_pad]
        #     print(norms.shape)
        #     angles = norm_angle[1:, random_pad]
        #     print(angles.shape)
        #     re_random_pad = np.random.randint(norm_angle.shape[0]-1, size=limit_dims)
        #     angles = angles[re_random_pad, :]
        #     print(angles.shape)
        #
        # else:
        #     norms = norm_angle[0, :]
        #     angles = norm_angle[1:, :]

        # norms = norm_angle[0, :]
        # angles = norm_angle[1:, :]
        #
        # print(norms)
        # print(angles)

        # angles = np.triu(angles, k=1).flatten()
        angles = angles.flatten()

        batch_sizes_sweeps[title] = (norms, angles)


    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Angle and norm dispersion in SGD')

    for i, (batch_size, stats) in enumerate(batch_sizes_sweeps.items()):
        norms, angles = stats

        norms_d = gaussian_kde(norms)
        xs = np.linspace(np.min(norms), np.max(norms))
        norms_d.covariance_factor = lambda: .25
        norms_d._compute_covariance()
        ax1.plot(xs, norms_d(xs),
                 color=comparison_colors[i],
                 label=batch_size)


        angles_d = gaussian_kde(angles)
        xs = np.linspace(np.min(angles), np.max(angles))
        angles_d.covariance_factor = lambda: .25
        angles_d._compute_covariance()
        ax2.plot(xs, angles_d(xs),
                 color=comparison_colors[i],
                 label=batch_size)


    # ax1.axvline(x=0, color='r')
    # ax2.axvline(x=0, color='r')

    ax1.legend()

    ax1.set_ylabel('updates norm % (d)')
    ax2.set_ylabel('updates angles (d)')

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax2.set_ylim(bottom=1e-6)

    ax1.set_xlabel('L2 norm of batch update tensor')
    ax2.set_xlabel('deg')

    fig.set_dpi(160)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    colors_locked = False
    active_cmap = 'cividis'  # 'cividis' for colorblind version  # 'Dark2' for pretty
    # absolute path of the directory where the run records were saved
    _runs_save_directory = "TO PROVIDE: absolute path to the /run/ directory"

    _runs_save_directory = Path(_runs_save_directory)

    generate_dim_sampling_offset_sweep()

    _experiments = parse_flatness_experiments(_runs_save_directory)
    normalize_experiments(_experiments)
    inspect_experiments(_experiments)

    compare_flatness(_experiments,
                     15,
                     [17, 16],
                     explicit_title="Minima flatness:\n dropout & batch size",
                     forced_base_legend='reference',
                     forced_legends=['-DO Bx16 Ex2', '-DO Bx16'])

    compare_flatness(_experiments,
                     4,
                     [5, 6, 7],
                     explicit_title="Minima flatness:\n dropout",
                     forced_base_legend='DO/DO:.0',
                     forced_legends=['DO/DI:.1', 'DO/DI:.25', 'DO/DI:.5'])

    compare_flatness(_experiments,
                     5,
                     [9, 10, 11, 12],
                     explicit_title="Minima flatness:\n feature size",
                     forced_base_legend='reference',
                     forced_legends=['LM/2 LW/2',
                                     'LW/2',
                                     'LMx2',
                                     'LMx4 LWx2'])

    compare_flatness(_experiments,
                     12,
                     [13, 14],
                     explicit_title="Minima flatness:\n dropout on large",
                     forced_base_legend='DO/DI:.1/.1',
                     forced_legends=['DO/DI:.25/.1',
                                     'DO/DI:.25/.05'])

    compare_flatness(_experiments,
                     21,
                     [22, 16],
                     explicit_title="Minima flatness:\n batch size",
                     forced_base_legend='B:4',
                     forced_legends=['B:32',
                                     'B:128'])

    compare_flatness(_experiments,
                     18,
                     [19, 20],
                     explicit_title="Minima flatness:\n epochs",
                     forced_base_legend='E:2',
                     forced_legends=['E:8',
                                     'E:32'])

    compare_flatness(_experiments,
                     23,
                     [24, 25],
                     explicit_title="Minima flatness:\n fine-tuning",
                     forced_base_legend='Robust Wide',
                     forced_legends=['Brittle Narrow',
                                     'Brittle Wide'])

    compare_robustness(_experiments,
                        23,
                        [24, 25],
                        explicit_title="Robustness and generalisation vs flatness",
                        forced_base_legend='Robust Wide',
                        forced_legends=['Brittle Narrow',
                                     'Brittle Wide'])

    compare_flatness(_experiments,
                     29,
                     [28, 27],
                     explicit_title="Minima flatness:\n Pre-fine tune",
                     forced_base_legend='Robust Wide',
                     forced_legends=['Brittle Narrow',
                                     'Brittle Wide'],
                     fine_tune_steps=0)

    compare_flatness(_experiments,
                 29,
                 [28, 27],
                 explicit_title="Minima flatness:\n fine tune epoch 1",
                 forced_base_legend='Robust Wide',
                 forced_legends=['Brittle Narrow',
                                 'Brittle Wide'],
                 fine_tune_steps=1)

    compare_flatness(_experiments,
                 29,
                 [28, 27],
                 explicit_title="Minima flatness:\n fine tune epoch 2",
                 forced_base_legend='Robust Wide',
                 forced_legends=['Brittle Narrow',
                                 'Brittle Wide'],
                 fine_tune_steps=2)

    compare_flatness(_experiments,
                 29,
                 [28, 27],
                 explicit_title="Minima flatness:\n fine tune epoch 3",
                 forced_base_legend='Robust Wide',
                 forced_legends=['Brittle Narrow',
                                 'Brittle Wide'],
                 fine_tune_steps=3)

    compare_flatness(_experiments,
             27,
             [27, 27, 27],
             explicit_title="Minima flatness:\n Brittle wide fine tunes",
             forced_base_legend='Pre Fine-Tune',
             forced_legends=['Fine-Tune epoch 1',
                             'Fine-Tune epoch 2',
                             'Fine-Tune epoch 3'],
             fine_tune_steps=0,
             fine_tune_sweep=True)

    compare_fine_tune_speed(_experiments,
                            29,
                            [28, 27],
                            explicit_title="Fine Tuning speed",
                            forced_base_legend='Robust Wide',
                            forced_legends=['Brittle Narrow',
                                            'Brittle Wide'],
                            assumed_fine_tunes=3)

    go_ea_net = 'CompressedNet_C0PFYDAGQ3'
    go_ea_sweep_dir = "2022-01-27_13h-25m-45s/"
    go_ea_sweep_variations = {
        'EDx8': '_ft_SNUR',
        'ED/8': '_ft_GJMR',
        'POPx2': '_ft_PDAF',
        'POP/5': '_ft_J3G8'
    }
    go_ea_sweep_main_trunk_dir = '2022-01-27_09h-37m-58s'
    go_ea_sweep_main_track = [
        '2022-01-26_10h-46m-57s',
        '2022-01-26_18h-25m-36s',
        '2022-01-26_19h-16m-29s',
        '2022-01-26_19h-57m-16s',
        '2022-01-27_00h-14m-16s',
        '2022-01-27_09h-37m-58s',
        ]

    parse_go_ea_experiments(_runs_save_directory,
                            go_ea_net,
                            go_ea_sweep_dir,
                            go_ea_sweep_variations,
                            go_ea_sweep_main_trunk_dir,
                            go_ea_sweep_main_track)

    circular_sweep_directory = '2022-01-27_16h-27m-49s'

    parse_circular_sweep_dirs(_runs_save_directory,
                              circular_sweep_directory,
                              tune_status_titles=['pre-train', 'pre-fine-tune', 'post-fine_tune']
                              )

    sgd_sweep_directory = [
        # '2022-08-14_01h-01m-31s',  # this folder is not provided, due to being ~ 4GB
                           '2022-08-14_00h-59m-18s',
                           '2022-08-14_01h-17m-23s',
                           '2022-08-14_01h-18m-23s'
    ]
    sgd_sweep_annotation = [
        # 'batch of 4',   # this folder is not provided, due to being ~ 4 GB
                            'batch of 32',
                            'batch of 128',
                            'batch of 1024'
    ]

    parse_sgd_update_tensor_dirs(_runs_save_directory,
                                 sgd_sweep_directory,
                                 sgd_sweep_annotation)
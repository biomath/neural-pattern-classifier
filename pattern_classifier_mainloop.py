__author__ = 'Matheus'
__version__ = '03/25/19'

import logging
from random import shuffle, choice
from itertools import chain
from copy import deepcopy
import time as t
import csv
from datetime import date
import multiprocessing as mp
# from fasteners import InterProcessLock

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, ttest_ind
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import resample

import process_spike_data as psd
import pattern_classifier_functions as pcf


def stim_classify(rasters_dict,
                  method,
                  brain_area,
                  count_bin=None,
                  refractory_period=0.001,
                  downsample_q=None,
                  dtw_radius=None,
                  trial_shuffling=False,
                  simulations=1000):
    """
    This function performs the bulk of the classification computation
    It grabs the rasterized spikes (0s and 1s), bins/filters them, and compares them
    If this is a trial shuffling run, it'll also create a list to store the accuracies of every simulation
    """
    number_of_stimuli = len(rasters_dict.keys())
    # print("Stim classify...")
    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    rasters_dict = dict((key_train[0], key_train[1][0:number_of_stimulus_repetitions])
                        for key_train in rasters_dict.items())

    if trial_shuffling is True:
        counter_diagonal_accuracies = np.zeros(simulations)

    if method in ('rcorr', 'dtw', 'cross_corr'):
        # Loop runs until all combinations of trains have been used as templates
        # Use template 1 only as a proxy for all
        def test_template(train, template_set):
            """
            This is a helper function that finds the best match in the rasters
            """
            def max_indices(lst):
                """
                This function returns the indices of all the maxima in the list
                """
                result = []
                offset = -1
                while True:
                    try:
                        offset = lst.index(max(lst), offset + 1)
                    except ValueError:
                        return result
                    result.append(offset)

            temp = list()
            if method == 'rcorr':
                for template in template_set:
                    # responses are rescaled to [0, 1] interval to minimize firing rate effects.
                    # This will be hardcoded for now
                    temp.append(pcf.corr_function(train, template, method, normalize=True))
            elif method == 'dtw':
                for template in template_set:
                    dtw_inv_distance = pcf.corr_function(train, template, method, normalize=True, dtw_radius=dtw_radius)
                    temp.append(dtw_inv_distance)
            elif method == 'cross_corr':
                for template in template_set:
                    temp.append(pcf.corr_function(train, template, method, normalize=True))

            index_max = max_indices(temp)

            # If there are ties, choose randomly
            if len(index_max) > 1:
                index_max = choice(index_max)

            return index_max

    else:
        def test_template(train, template_set):
            """
            This is a helper function that finds the best match in the rasters
            """
            def min_indices(lst):
                """
                This function returns the indices of all the minima in the list
                """
                result = []
                offset = -1
                while True:
                    try:
                        offset = lst.index(min(lst), offset + 1)
                    except ValueError:
                        return result
                    result.append(offset)

            temp = list()

            if method == 'count':
                for template in template_set:
                    # vector dot-subtraction (element-by-element, aka bin-by-bin)
                    #   then square all values (squared distances) then sum them
                    temp.append(np.sum(np.power(np.subtract(train, template), 2)))

            index_min = min_indices(temp)
            # If there are ties, choose randomly
            if len(index_min) > 1:
                index_min = choice(index_min)
            return index_min

    # Will hold the total average of classifications
    counter_accuracy_matrix = np.zeros([number_of_stimuli, number_of_stimuli])

    if downsample_q is not None:
        rasters_dict = dict((key_train[0], [resample(train, len(train)//downsample_q)
                                            for train in key_train[1]])
                            for key_train in rasters_dict.items())

    if method == 'count':
        rasters_dict = dict((key_train[0], [psd.bin_train(train, refractory_period, count_bin)
                                            for train in key_train[1]])
                            for key_train in rasters_dict.items())

    for i in np.arange(0, simulations):
        # Creates copies of raster dictionary to pop templates from it
        temp_raster_dict = deepcopy(rasters_dict)

        template_list = list()
        for key in rasters_dict.keys():
            template_list.append(temp_raster_dict[key].pop(np.random.randint(0, number_of_stimulus_repetitions)))

        # Will hold accuracies in the current run of the classifier
        # rows are actual stimulus, columns are predicted
        current_matrix = np.zeros([number_of_stimuli, number_of_stimuli])

        # Take responses to each stimuli and compare to set of templates
        actual_stim_index = 0
        for key in rasters_dict.keys():
            for train in temp_raster_dict[key]:
                predicted_stim_index = test_template(train, template_list)
                current_matrix[actual_stim_index, predicted_stim_index] += 1
            actual_stim_index += 1

        # Sum performance of current templates and add to main matrix
        counter_accuracy_matrix += current_matrix

        if trial_shuffling is True:
            if brain_area == 'HVC':
                # Only one BOS
                counter_diagonal_accuracies[i] = current_matrix[0, 0]

            elif brain_area == 'NCM':
                # Compute means
                counter_diagonal_accuracies[i] = np.mean(np.diag(current_matrix))
            else:
                logging.info("Problem with brain area parameter!")
                return

    # Transform values from accuracy counts to averages
    # number_of_stimulus_repetitions - 1 explanation: one template is left out,
    #   so N - 1 comparisons are made per stimulus
    mean_accuracy_matrix = counter_accuracy_matrix / (simulations * (number_of_stimulus_repetitions - 1))
    mean_diagonal_accuracies = counter_diagonal_accuracies / (number_of_stimulus_repetitions - 1)

    if trial_shuffling is True:
        return mean_accuracy_matrix, mean_diagonal_accuracies
    else:
        return mean_accuracy_matrix


def trial_shuffling_approach(rasters_dict, mapped_accuracies, method, brain_area, count_bin=(2,),
                             simulations=1000, downsample_q=None, dtw_radius=1):
    """
    This function is a modification of the algorithm designed by 
        Caras et al. The Journal of Neuroscience 35(8): 3431–3445, 2015
    It jumbles the data and reclassifies it. 
    The accuracy values should be centered around 100/number_of_stimulus_repetitions 
    """

    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    # Making a flat list out of the dictionary
    jumbled_rasters = list(chain.from_iterable(rasters_dict.values()))

    shuffle(jumbled_rasters)

    number_of_stimuli = len(rasters_dict)
    # Randomization
    jumbled_rasters_dict = dict((key, []) for key in rasters_dict.keys())
    dict_keys = list(jumbled_rasters_dict.copy().keys())
    for i, train in enumerate(jumbled_rasters):
        jumbled_rasters_dict[dict_keys[i % number_of_stimuli]].append(train)  # distribute trains among keys of the dict

    # Classification
    _, jumbled_accuracies = stim_classify(jumbled_rasters_dict,
                                          brain_area=brain_area,
                                          method=method,
                                          count_bin=count_bin,
                                          downsample_q=downsample_q,
                                          dtw_radius=dtw_radius,
                                          trial_shuffling=True,
                                          simulations=simulations)

    jumbled_CI = norm.interval(0.95, loc=np.mean(jumbled_accuracies),
                               scale=np.std(jumbled_accuracies, ddof=1))

    ci_significance = np.mean(mapped_accuracies) > jumbled_CI[1]

    # Returns mapped accuracies again (kindda dumb)

    # Two-tailed t_test with Welch's correction
    t_value, p_two_tailed = ttest_ind(mapped_accuracies, jumbled_accuracies, equal_var=False)
    
    # Cohen d approximation from difference of means over sum of standard deviations
    cohen_d = \
        (np.mean(mapped_accuracies) - np.mean(jumbled_accuracies)) / \
        np.sqrt((np.std(mapped_accuracies, ddof=1) ** 2 + np.std(jumbled_accuracies, ddof=1) ** 2) / 2)

    # Getting the one-tailed p value from a two-tailed test:
    # If T is in the hypothesis direction (t > 0 in this case): one-tailed p is half of the two-tailed
    # Elif T is in the contrary direction: one-tailed p is 1 minus half of the two-tailed
    if t_value >= 0:
        p_one_tailed = p_two_tailed / 2
    else:
        p_one_tailed = 1 - p_two_tailed / 2

    return t_value, p_one_tailed, cohen_d, ci_significance, jumbled_accuracies, mapped_accuracies


def write_data_to_csv(matrix, unit_name, method, stim_list, output_path,
                      significance=None, t_value=None, p_value=None,
                      cohen_d=None, best_sigma=None, best_bin=None):


    df = pd.DataFrame(matrix, columns=stim_list,
                      index=stim_list)

    with open(str(output_path + "\\" + unit_name + '.csv'), 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows([[unit_name],
                          ['Analysis method: ' + method],
                          ['Best sigma = ' + str(best_sigma)],
                          ['Best bin = ' + str(best_bin)],
                          ['Mean > 95% value of jumbled = ' + str(significance)],
                          ['One tailed T-test statistic (> jumbled spikes) = ' + str(t_value)],
                          ['P value = ' + str(p_value)],
                          ['Cohen\'s d = ' + str(cohen_d)]])
        df.to_csv(file)


def run_classification(memory_name,
                       method,
                       refractory_period,
                       pre_stimulus_time, post_stimulus_time,
                       pre_stimulus_raster, post_stimulus_raster,
                       brain_area,
                       accuracies_master_sheet_fullpath,
                       statistics_master_sheet_fullpath,
                       unit_name,
                       number_of_simulations,
                       output_path,
                       count_bin_set=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
                       master_sheet_columns=(),
                       key_name=None,
                       dtw_radius=None,
                       downsample_q=None,
                       sigma_set=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                       write_master_sheet=True,
                       save_plot=True,
                       save_rasters=True,
                       save_histograms=True,
                       write_to_csv=True, file_name=None, pdf_handle=None):
    """
    This function is the main loop for the classification algorithm.
    It keeps track of the methods employed and flow: classification -> trial shuffling -> CSV writing / plotting
    It also outputs a log to verify runtime errors and where the code crashed
    """
    from re import split
    date_today = str(date.today())
    date_today = date_today[:4] + date_today[5:7] + date_today[8:10]  # Taking hyphens out
    logging.basicConfig(format='%(asctime)s %(processName)s %(message)s',
                        filename=mp.current_process().name + '_' + date_today + '.log',
                        level=logging.INFO)

    if key_name is None:
        key_name = memory_name

    def key_with_max_val(dictionary):
        """
        a) create a list of the dict's keys and values;
        b) return the key with the max value
        """
        v = list(dictionary.values())
        k = list(dictionary.keys())
        return k[v.index(max(v))]

    best_bin = None
    best_sigma = None

    if method in ('rcorr', 'cross_corr', 'dtw'):
        logging.info('Initializing spike timing classification for ' + memory_name + '...')

        t0 = t.time()

        peristim_dict = psd.load_rasters(memory_name, key_name=key_name, refractory_period=refractory_period,
                                         pre_stimulus_time=pre_stimulus_time,
                                         post_stimulus_time=post_stimulus_time)
        if peristim_dict is not None:
            # Not worth running when cell fires <1 spike
            number_of_stimuli = len(peristim_dict.keys())

            temp_matrix_dict = {}
            temp_mapped_accuracies_dict = {}
            mean_of_accuracies_dict = {}
            for current_sigma in sigma_set:
                peristim_dict = psd.spike_timing_process_data(peristim_dict, refractory_period=refractory_period,
                                                              sigma=current_sigma)

                temp_matrix_dict[current_sigma], temp_mapped_accuracies_dict[current_sigma] = \
                    stim_classify(peristim_dict,
                                  method=method,
                                  dtw_radius=dtw_radius,
                                  downsample_q=downsample_q,
                                  trial_shuffling=True,
                                  simulations=number_of_simulations,
                                  brain_area=brain_area)
                # Sum of the sums of the accuracies for getting an accuracy measure
                if brain_area == 'NCM':
                    mean_of_accuracies_dict[current_sigma] = np.mean(np.diag(temp_matrix_dict[current_sigma]))
                else:
                    mean_of_accuracies_dict[current_sigma] = temp_matrix_dict[current_sigma][0, 0]

            best_sigma = key_with_max_val(mean_of_accuracies_dict)
            accuracy_matrix = temp_matrix_dict[best_sigma]
            mapped_accuracies = temp_mapped_accuracies_dict[best_sigma]  # For trial shuffling
            # Alter peristim_dict to become a gaussian using the best sigma
            peristim_dict = psd.spike_timing_process_data(peristim_dict, refractory_period=refractory_period,
                                                          sigma=best_sigma)

            f = plot_performance(temp_mapped_accuracies_dict, number_of_stimuli=number_of_stimuli, method=method,
                             title=file_name + " " + method)
            pdf_handle.savefig()
            f.clear()
            plt.close()

            logging.info('Classification completed after ' + str(t.time() - t0) + ' seconds.')

    else:
        logging.info('Initializing spike count classification for ' + memory_name + '...')

        t0 = t.time()
        temp_matrix_dict = {}
        temp_mapped_accuracies_dict = {}  # For trial shuffling
        mean_of_accuracies_dict = {}

        peristim_dict = psd.load_rasters(memory_name, key_name=key_name, refractory_period=refractory_period,
                                         pre_stimulus_time=pre_stimulus_time,
                                         post_stimulus_time=post_stimulus_time)

        if peristim_dict is not None:
            # Not worth running when cell fires <1 spike

            number_of_stimuli = len(peristim_dict.keys())

            for current_bin in count_bin_set:
                if current_bin == 'Stim duration':
                    current_bin = post_stimulus_time - pre_stimulus_time
                temp_matrix_dict[current_bin], temp_mapped_accuracies_dict[current_bin] = \
                    stim_classify(peristim_dict,
                                  method=method,
                                  count_bin=current_bin,
                                  refractory_period=refractory_period,
                                  dtw_radius=dtw_radius,
                                  downsample_q=downsample_q,
                                  trial_shuffling=True,
                                  simulations=number_of_simulations,
                                  brain_area=brain_area)

                # Sum of the sums of the accuracies for getting an accuracy measure
                if brain_area == 'NCM':
                    mean_of_accuracies_dict[current_bin] = np.mean(np.diag(temp_matrix_dict[current_bin]))
                else:
                    mean_of_accuracies_dict[current_bin] = temp_matrix_dict[current_bin][0, 0]

            best_bin = key_with_max_val(mean_of_accuracies_dict)
            accuracy_matrix = temp_matrix_dict[best_bin]
            mapped_accuracies = temp_mapped_accuracies_dict[best_bin]  # For trial shuffling

            f = plot_performance(temp_mapped_accuracies_dict, number_of_stimuli=number_of_stimuli, method=method,
                                 title=file_name + " " + method)
            pdf_handle.savefig()
            f.clear()
            plt.close(f)

            logging.info('Classification completed after ' + str(t.time() - t0) + ' seconds.')

    if peristim_dict is not None:
        logging.info('Initializing trial shuffling...')
        t0 = t.time()
        # Trial shuffling

        t_value, p_one_tailed, cohen_d, ci_significance, jumbled, mapped = \
            trial_shuffling_approach(peristim_dict, mapped_accuracies, count_bin=best_bin,
                                     downsample_q=downsample_q, method=method, brain_area=brain_area,
                                     simulations=number_of_simulations)

        logging.info('TSA completed after ' + str(t.time() - t0) + ' seconds.')
        # Plots

        if file_name is None:
            file_name = memory_name

        if save_histograms is True:
            f = plot_trial_shuffling_histogram(jumbled, mapped, title=file_name + " " + method)
            pdf_handle.savefig()
            f.clear()
            plt.close(f)


        # Acquire stim names
        stim_names = list(peristim_dict.keys())

        if save_plot is True:
            logging.info('Plotting confusion matrices...')
            f = plot_cm(accuracy_matrix, stim_names, title=file_name + " " + method)
            pdf_handle.savefig()
            f.clear()
            plt.close(f)

        if save_rasters is True:
            logging.info('Plotting rasters...')
            f = plot_rasters(memory_name, title=file_name + " " + method, key_name=key_name, refractory_period=refractory_period,
                             pre_stimulus_raster=pre_stimulus_raster, post_stimulus_raster=post_stimulus_raster,
                             post_stimulus_time=post_stimulus_time, method=method, downsample_q=downsample_q,
                             sigma=best_sigma, stim_names=stim_names)
            pdf_handle.savefig()
            f.clear()
            plt.close(f)

        # For the purposes of the write_to_csv function
        if method == 'count':
            best_sigma = None

        if write_to_csv is True:
            logging.info('Writing to CSV...')

            write_data_to_csv(accuracy_matrix, unit_name=unit_name, output_path=output_path, method=method,
                              stim_list=stim_names,
                              t_value=t_value, p_value=p_one_tailed, cohen_d=cohen_d,
                              significance=ci_significance,
                              best_sigma=best_sigma, best_bin=best_bin)

    else:
        # This is just to log a useless cell as NA values in the mastersheet
        t_value = p_one_tailed = cohen_d = ci_significance = best_sigma = best_bin = np.NaN
        accuracy_matrix = np.empty((4, 4,))
        number_of_stimuli = len(accuracy_matrix[0])
        accuracy_matrix[:] = np.NaN
        stim_names = [np.NaN]


    # Headings:
    # ['Unit'] + ['Hemisphere'] + ['Method'] + ['Trial'] + ['T.statistic'] + ['P.value'] + ['Cohen.d'] +
    # ['CI.significance'] + ['Sigma'] +
    # ['Stimulus'] + ['Accuracy']
    if write_master_sheet:

        # lock = InterProcessLock(accuracies_master_sheet_fullpath)  # I'm getting Permission Denied with this...
        # if lock.acquire(blocking=True):
        with open(accuracies_master_sheet_fullpath, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i, stimulus in enumerate(stim_names):
                writer.writerow(
                    [unit_name] + [split("\\\\", key_name)[-1][:-4]] + [method] +
                    [item for item in master_sheet_columns] +
                    [stimulus] + [accuracy_matrix[i, i]])

        # lock = InterProcessLock(statistics_master_sheet_fullpath)
        # if lock.acquire(blocking=True):
        with open(statistics_master_sheet_fullpath, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(
                [unit_name] + [split("\\\\", key_name)[-1][:-4]] + [method] +
                [item for item in master_sheet_columns] +
                [t_value] + [p_one_tailed] + [cohen_d] +
                [ci_significance] + [best_sigma] + [best_bin])

    plt.close("all")


def plot_trial_shuffling_histogram(jumbled_data, mapped_data, method=None, title=None):
    f = plt.figure()
    f.set_size_inches((24, 13.3))
    ax = f.add_subplot(111)
    # Transform accuracy measures to percentages
    jumbled_data_transformed = np.array(jumbled_data) * 100
    mapped_data_transformed = np.array(mapped_data) * 100

    xs = np.linspace(0, 100, 2000)

    try:
        jumbled_density = gaussian_kde(jumbled_data_transformed)
    except np.linalg.linalg.LinAlgError as err:
            logging.info(str(err) + "; Skipping trial shuffling")
            return

    jumbled_density.covariance_factor = lambda: .2
    jumbled_density._compute_covariance()
    jumbled_CI_x = (norm.interval(0.95, loc=np.mean(jumbled_data_transformed),
                                  scale=np.std(jumbled_data_transformed, ddof=1))[1])

    shuffled_plot, = ax.plot(xs, jumbled_density(xs), label='Shuffled', color='k')
    ax.fill_between(xs, 0, jumbled_density(xs), facecolor='k', alpha=0.3)

    CI_line = ax.axvline(jumbled_CI_x, color='k', lw=2, ls='--')

    mapped_mean = np.mean(mapped_data_transformed)
    if np.var(mapped_data_transformed) != 0:
        mapped_density = gaussian_kde(mapped_data_transformed)
        mapped_density.covariance_factor = lambda: .2
        mapped_density._compute_covariance()
        mapped_plot, = ax.plot(xs, mapped_density(xs), label='Original', color='b')
        ax.fill_between(xs, 0, mapped_density(xs), facecolor='b', alpha=0.3)
    else:
        mapped_plot, = ax.plot(2000, 0, label='Original', color='b')  # invisible dot

    mean_line = ax.axvline(mapped_mean, color='b', lw=2, ls=':')

    ax.set_xlim([0, 100])

    f.legend([shuffled_plot, CI_line, mapped_plot, mean_line],
             ["Shuffled", "Shuffled 95% CI", "Original", "Original mean"], frameon=False)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Frequency of values")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if title is not None:
        f.suptitle(title)

    plt.tight_layout()

    return f


def plot_rasters(memory_name, refractory_period, pre_stimulus_raster, post_stimulus_raster, post_stimulus_time, title,
                 method, downsample_q=None, key_name=None, sigma=None, stim_names=None, pdf_handle=None):
    """
    :param memory_name:
    :param refractory_period:
    :param pre_stimulus_raster:
    :param post_stimulus_raster:
    :param post_stimulus_time:
    :param key_name:
    :param sigma:
    :param stim_names:
    :return:
    """

    rasters_dict = psd.load_rasters(memory_name, refractory_period, pre_stimulus_raster,
                                    post_stimulus_raster,
                                    key_name=key_name)

    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    number_of_stimuli = len(rasters_dict.keys())

    if sigma is not None:
        rasters_dict = psd.spike_timing_process_data(
            rasters_dict, refractory_period=refractory_period, sigma=sigma)

    f, axarr = plt.subplots(number_of_stimulus_repetitions, number_of_stimuli, sharex='col', sharey='row')
    f.set_size_inches((24, 13.3))

    # Populate with data
    for plot_row_idx in np.arange(0, number_of_stimulus_repetitions):
        for plot_col_idx, stim_name in enumerate(rasters_dict.keys()):
            if method != 'count' and downsample_q is not None:
                train_to_plot = resample(rasters_dict[stim_name][plot_row_idx],
                                         len(rasters_dict[stim_name][plot_row_idx]) // downsample_q)
            else:
                train_to_plot = rasters_dict[stim_name][plot_row_idx]

            if method != 'count':
                axarr[plot_row_idx, plot_col_idx].plot(
                    np.linspace(-pre_stimulus_raster, post_stimulus_raster, train_to_plot.size),
                    train_to_plot,
                    color='black')
            else:
                train_to_plot[train_to_plot == 0.0] = np.nan
                axarr[plot_row_idx, plot_col_idx].plot(
                    np.linspace(-pre_stimulus_raster, post_stimulus_raster, train_to_plot.size),
                    train_to_plot, "|",
                    markersize=mpl.rcParams['font.size'],  # Change from default on purpose. 1.5 is the scale
                    markeredgewidth=mpl.rcParams['font.size'] / 100,
                    color='black', alpha=0.8)
                # axarr[plot_row_idx, plot_col_idx].set_ylim([0.9, 1.1])

            axarr[plot_row_idx, plot_col_idx].axvspan(xmin=0, xmax=post_stimulus_time, color='black', alpha=0.2)

            axarr[plot_row_idx, plot_col_idx].axis('off')

    for idx, stim_name in enumerate(stim_names):
        axarr[0, idx].set_title(stim_name)

    plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
    # plt.setp([a.get_xticklabels() for a in axarr[number_of_stimulus_repetitions - 1, :]])

    f.suptitle(title)

    plt.subplots_adjust(hspace=0)

    return f


def plot_cm(accuracy_matrix, stim_names, title):
    f = plt.figure()
    f.set_size_inches((24, 13.3))
    ax = f.add_subplot(1, 1, 1)
    cax = ax.matshow(accuracy_matrix * 100, interpolation='nearest', vmin=0, vmax=100, cmap='jet')
    ax.set_xticklabels([''] + stim_names)
    ax.set_yticklabels([''] + stim_names)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    f.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    return f


def plot_performance(mapped_accuracies_dict, method, number_of_stimuli, title=None):
    f = plt.figure()
    f.set_size_inches((24, 13.3))
    ax = f.add_subplot(111)

    if method == 'rcorr':
        color = 'navy'
    elif method == 'cross_corr':
        color = 'forestgreen'
    elif method == 'dtw':
        color = 'darkmagenta'
    else:
        color = 'firebrick'

    temp_items_list = sorted(list(mapped_accuracies_dict.items()))
    x_values = [key_value[0] for key_value in temp_items_list]
    y_values = [key_value[1] for key_value in temp_items_list]

    # Mean and 95% CI of each sigma/bin
    y_means = np.mean(y_values, axis=1) * 100
    y_cis = norm.interval(0.95, loc=np.mean(y_values, axis=1),
                          scale=np.std(y_values, axis=1, ddof=1))
    y_cis_range = np.subtract(y_cis[1], y_cis[0]) * 100

    cax = ax.errorbar(x_values, y_means, yerr=y_cis_range, fmt='-o', color=color, clip_on=False)
    for b in cax[1]:
        b.set_clip_on(False)

    def adjust_spines(ax, spines):
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward', 20))  # outward by 10 points
                spine.set_smart_bounds(True)
            else:
                spine.set_color('none')  # don't draw spine

        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])

    random_prob_line = 100 / number_of_stimuli
    ax.axhline(random_prob_line, color='k', lw='2', ls='--', alpha=0.2)
    ax.set_xscale("log", basex=2)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # ax.set_xlim([ax.get_xlim()[0]-1, ax.get_xlim()[1]+1])

    # xticklabels = [tup[0] for tup in temp_items_list]
    # PERFORMANCE_AX.set_xticklabels(xticklabels)
    #
    # ax.xaxis.get_major_ticks()[-1].label1.set_visible(False)
    # ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.set_xlabel('Bin/sigma size')
    ax.set_ylabel('Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    adjust_spines(ax, ['left', 'bottom'])

    plt.title(title)

    # f.show()
    plt.tight_layout()

    return f


def run_mi(memory_name,
           refractory_period,
           pre_stimulus_time, post_stimulus_time,
           bin_set=(2,),
           key_name=None):
    import time as t
    import csv
    def key_with_max_val(dictionary):
        """
        a) create a list of the dict's keys and values;
        b) return the key with the max value
        """
        v = list(dictionary.values())
        k = list(dictionary.keys())
        return k[v.index(max(v))]
    logging.info('Initializing spike count classification for ' + memory_name + '...')

    t0 = t.time()
    matrices_dicts = {}
    mi_dicts = {}  # For trial shuffling

    peristim_dict = psd.load_rasters(memory_name, key_name=key_name, refractory_period=refractory_period,
                                     pre_stimulus_time=pre_stimulus_time,
                                     post_stimulus_time=post_stimulus_time)
    for current_bin in bin_set:
        matrices_dicts[current_bin], mi_dicts[current_bin] = \
            pcf.mutual_information(peristim_dict,
                               bin=current_bin,
                               refractory_period=refractory_period)

    best_bin = key_with_max_val(mi_dicts)
    return best_bin, mi_dicts[best_bin]
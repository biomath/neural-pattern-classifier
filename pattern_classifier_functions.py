__author__ = 'Matheus'
__version__ = '03/25/19'

from random import choice
import numpy as np
from fastdtw import fastdtw
from copy import deepcopy
import process_spike_data as psd


def rcorr(g1, g2):
    """
    This function performs the RCorr computation between two smoothed filtered spike trains
    As described in Schreiber et al. Neurocomputing 52-54: 925–931, 2003.
    Modifications for when vectors are 0 were added to support for low-firing neurons
    """
    # When correlating two flat responses, rcorr yields 'nan'; this "corrects" it
    if (np.sum(g1) == 0) and (np.sum(g2) == 0):
        return np.nan
    elif (np.sum(g1) == 0) or (np.sum(g2) == 0):
        return np.nan
    else:
        return np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))


def normalize_vector(g):
    """
    Vector is rescaled to a [0, 1] interval
    """
    if np.sum(g) != 0:
        return (g - np.min(g)) / (np.max(g) - np.min(g))
    else:
        return g


def cross_corr(g1, g2):
    """
    This function performs the cross correlation computation
    Modifications for when vectors are 0 were added to support for low-firing neurons
    """
    if (np.sum(g1) == 0) and (np.sum(g2) == 0):
        return 1
    elif (np.sum(g1) == 0) or (np.sum(g2) == 0):
        return 0
    else:
        return np.corrcoef(g1, g2)[1, 0]


def dtw_nearness(g1, g2, radius):
    """
    This function performs the fast dynamic time warping algorithm
    As described in Salvador & Chan. Intelligent Data Analysis 11: 561-580, 2007
    Modifications for when vectors are 0 were added to support for low-firing neurons
    requires pip install fastdtw
    """

    ret_val, _ = fastdtw(g1, g2, radius=radius)
    try:
        ret_val = 1.0 / ret_val
    except ZeroDivisionError:
        ret_val = 1
    return ret_val


def corr_function(g1, g2, correlation_method, dtw_radius=1, normalize=True):
    """
    Wrapper function to direct code to the correct distance method
    """
    if correlation_method == 'count':
        return np.absolute(np.sum(g1) - np.sum(g2))
    else:
        if normalize:
            g1 = normalize_vector(g1)
            g2 = normalize_vector(g2)
        if correlation_method == 'rcorr':
            return rcorr(g1, g2)
        elif correlation_method == 'cross_corr':
            return cross_corr(g1, g2)
        elif correlation_method == 'dtw':
            ret_val, _ = fastdtw(g1, g2, radius=dtw_radius)
            return ret_val


def mutual_information(rasters_dict,
                       bin,
                       refractory_period=0.001):
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

    rasters_dict = dict((key_train[0], [psd.bin_train(train, refractory_period, bin)
                                        for train in key_train[1]])
                        for key_train in rasters_dict.items())

    def test_template(test_train, remaining_trains_dict):
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

        min_distances_per_stim = list()
        for key in sorted(list(remaining_trains_dict.keys())):
            temp_distances_per_stim = list()
            for remaining_train in remaining_trains_dict[key]:
                # vector dot-subtraction (element-by-element, aka bin-by-bin)
                #   then square all values (squared distances) then sum them
                temp_distances_per_stim.append(np.sqrt(np.sum(np.power(np.subtract(test_train, remaining_train), 2))))
            min_distances_per_stim.append(temp_distances_per_stim[min_indices(temp_distances_per_stim)[0]])

        index_min = min_indices(min_distances_per_stim)
        # If there are ties, choose randomly
        if len(index_min) > 1:
            index_min = choice(index_min)
        return index_min

    current_matrix = np.zeros([number_of_stimuli, number_of_stimuli])
    actual_stim_index = 0
    for key in sorted(list(rasters_dict.keys())):

        for dummy_idx, train in enumerate(rasters_dict[key]):
            # Creates copy of raster dictionary to pop trains from it
            temp_raster_dict = deepcopy(rasters_dict)
            del temp_raster_dict[key][dummy_idx]
            predicted_stim_index = test_template(train, temp_raster_dict)
            current_matrix[actual_stim_index, predicted_stim_index] += 1
        actual_stim_index += 1

    current_matrix /= (number_of_stimulus_repetitions - 1.)

    # My take on MI. Need to check if it's correct:
        # MI = sum of p(r, s) * log2 {p(r, s)/[p(r)*p(s)]}, where:
            # p(r, s) = p(r|s)*p(s): classification accuracy for a given stimulus s multiplied by p(s)
            # p(r) = 1/stimulus_repetitions (probability of classifying response as any given stimulus. Unsure about that...
            # p(s) = 1/number_of_stimuli
            # So it comes down to: sum of ( accuracy*p(s) * log2 {accuracy / p(r)} )
    mi = 0
    prob_stim = 1./number_of_stimuli
    for index in range(0, number_of_stimuli):
        curr_accuracy = current_matrix[index, index]
        curr_result = curr_accuracy * prob_stim * np.log2(curr_accuracy*number_of_stimulus_repetitions)
        if np.isnan(curr_result):
            curr_result = 0
        mi += curr_result

    return current_matrix, mi



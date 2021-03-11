from math import floor
import pandas as pd
import mne
import numpy as np

import dpm
import umne


class fn_template:
    epochs_full = dpm.consts.rsa_data_path + "epochs/full-{}/{}-epo.fif"

    @staticmethod
    def dissim(subdir, metric, locked_on_response, subj_id):
        return dpm.consts.rsa_data_path + \
               "dissim/{}/dissim_{}_{}_lock{}.dmat".format(subdir, metric, subj_id, 'response' if locked_on_response else 'stimulus')


#==============================================================================================================
# Compute dissimilarity
#==============================================================================================================

#-------------------------------------------------------------------------------------------
def preprocess_and_compute_dissimilarity_stim(subj_id, metrics, meg_channels=True, tmin=-0.2, tmax=1.0, baseline=(None, 0),
                                              lopass_filter=None, rejection='default', split_half=True, include_4=True):
    """
    Preprocess RSA data: load the epochs, and compute basic dissimilarity matrices.
    This is done while locking the epochs on the stimulus

    :param split_half: For the within-task dissimilarity, split the data in half
    :param include_4: whether to include stimuli with the digit 4 or not
    """

    if isinstance(metrics, str):
        metrics = metrics,

    epoch_filter = 'target > 10' if include_4 else 'target > 10 and decade != 4 and unit != 4'

    params = dict(meg_channels=meg_channels, tmin=tmin, tmax=tmax, baseline=baseline, lopass_filter=lopass_filter, rejection=rejection,
                  on_response=False, split_half=False, epoch_filter=epoch_filter)

    #-- Compute dissimilarity without split-half

    epochs_rsvp = _load_and_average(subj_id, dpm.consts.rsvp_raw_files, 'RSVP', baseline, lopass_filter, meg_channels, rejection,
                                    tmin=tmin, tmax=tmax, on_response=False, epoch_filter=epoch_filter, load_error_trials=True)

    epochs_comp = _load_and_average(subj_id, dpm.consts.comparison_raw_files, 'comp', baseline, lopass_filter, meg_channels, rejection,
                                    tmin=tmin, tmax=tmax, on_response=False, epoch_filter=epoch_filter, load_error_trials=False)

    for metric in metrics:
        _compute_and_save_dissimilarity(epochs_comp, epochs_rsvp, 'inter-task', subj_id, metric, False, params)

    #-- Compute within-task dissimilarity
    params['split_half'] = split_half

    if split_half:
        # noinspection PyUnusedLocal
        epochs_rsvp, epochs_comp = None, None  # cleanup memory
        epochs_rsvp1, epochs_rsvp2 = \
            splithalf_and_average(metadata_fields=('target', 'location'), subj_id=subj_id, data_filenames=dpm.consts.rsvp_raw_files,
                                  epoch_filter=epoch_filter, tmin=tmin, tmax=tmax, rejection=None, load_error_trials=True)

        epochs_comp1, epochs_comp2 = \
            splithalf_and_average(metadata_fields=('target', 'location'), subj_id=subj_id, data_filenames=dpm.consts.comparison_raw_files,
                                  epoch_filter=epoch_filter, tmin=tmin, tmax=tmax, rejection=None, load_error_trials=False)

    else:
        epochs_rsvp1, epochs_rsvp2 = epochs_rsvp, epochs_rsvp
        epochs_comp1, epochs_comp2 = epochs_comp, epochs_comp

    for metric in metrics:
        _compute_and_save_dissimilarity(epochs_rsvp1, epochs_rsvp2, 'full-RSVP', subj_id, metric, False, params)
        _compute_and_save_dissimilarity(epochs_comp1, epochs_comp2, 'full-comp', subj_id, metric, False, params)

    print('Saved.')


#-------------------------------------------------------------------------------------------
def preprocess_and_compute_dissimilarity_response(subj_id, metrics, meg_channels=True, tmin=-1.0, tmax=0.2,
                                                  lopass_filter=None, rejection='default', split_half=True, include_4=False):
    """
    Preprocess RSA data: load the epochs, and compute basic dissimilarity matrices.
    This is done while locking the epochs on the responses (so it's possible only for the comparison task)

    :param split_half: For the within-task dissimilarity, split the data in half
    :param include_4: whether to include stimuli with the digit 4 or not
    """

    if isinstance(metrics, str):
        metrics = metrics,

    epoch_filter = 'target > 10' if include_4 else 'target > 10 and decade != 4 and unit != 4'

    params = dict(meg_channels=meg_channels, tmin=tmin, tmax=tmax, baseline=None, lopass_filter=lopass_filter, rejection=rejection,
                  on_response=True, split_half=split_half, epoch_filter=epoch_filter)

    if split_half:
        epochs_comp1, epochs_comp2 = \
            splithalf_and_average(metadata_fields=('target', 'location'), subj_id=subj_id, data_filenames=dpm.consts.comparison_raw_files,
                                  epoch_filter=epoch_filter, tmin=tmin, tmax=tmax, rejection=None, on_response=True, load_error_trials=False)

    else:
        epochs_comp = _load_and_average(subj_id, dpm.consts.comparison_raw_files, subdir='comp', baseline=None, lopass_filter=lopass_filter,
                                        meg_channels=meg_channels, rejection=rejection, tmin=tmin, tmax=tmax,
                                        on_response=True, epoch_filter=epoch_filter, load_error_trials=False)

        epochs_comp1, epochs_comp2 = epochs_comp, epochs_comp

    for metric in metrics:
        _compute_and_save_dissimilarity(epochs_comp1, epochs_comp2, 'full-comp', subj_id, metric, True, params)

    print('Saved.')


#----------------------------------------------------
def _load_and_average(subj_id, filenames, subdir, baseline, lopass_filter, meg_channels, rejection, tmin, tmax, on_response, epoch_filter,
                      load_error_trials):

    epochs = dpm.files.load_subj_epochs(subj_id, filenames, lopass_filter=lopass_filter, meg_channels=meg_channels, tmin=tmin, tmax=tmax,
                                        baseline=baseline, rejection=rejection, on_response=on_response, load_error_trials=load_error_trials)
    epochs = epochs[epoch_filter]
    avg = umne.epochs.average_epochs_by_metadata(epochs, ('target', 'location'))
    avg.save(fn_template.epochs_full.format(subdir, subj_id), overwrite=True)

    return avg


#----------------------------------------------------
def _compute_and_save_dissimilarity(epochs1, epochs2, subdir, subj_id, metric, locked_on_response, params):

    print('\n\nComputing {} dissimilarity (metric={})...'.format(subdir, metric))

    dissim = umne.rsa.gen_observed_dissimilarity(epochs1, epochs2, metric=metric, sliding_window_size=100, sliding_window_step=10)

    filename = fn_template.dissim(subdir, metric, locked_on_response, subj_id)
    print('Saving the dissimilarity matrix to {}'.format(filename))
    dissim.save(filename)
    with open(filename + "_params.txt", 'w') as fp:
        fp.write(str(params) + '\n')


#----------------------------------------------------------------------------------------------------
def load_splithalf_and_compute_dissimilarity(filename, n_pca=30, metric='spearmanr', grouping_metadata_fields=None,
                                             grouping_metadata_fields1=None,
                                             sliding_window_size=None, sliding_window_step=None, sliding_window_min_size=None,
                                             debug=None):

    avg1, avg2 = load_average_epochs_splithalf(filename)

    if grouping_metadata_fields is not None:
        avg1 = umne.epochs.average_epochs_by_metadata(avg1, grouping_metadata_fields)
        grouping_metadata_fields1 = grouping_metadata_fields1 or grouping_metadata_fields
        avg2 = umne.epochs.average_epochs_by_metadata(avg2, grouping_metadata_fields1)

    dissim = umne.rsa.gen_observed_dissimilarity(avg1, avg2, n_pca=n_pca, metric=metric, sliding_window_size=sliding_window_size,
                                                 sliding_window_step=sliding_window_step, sliding_window_min_size=sliding_window_min_size,
                                                 debug=debug)

    return dissim


#----------------------------------------------------------------------------------------------------
def load_average_epochs_splithalf(filename):
    """
    Load previously-saved split-halved epochs files
    """

    avg1 = mne.read_epochs(filename + '_1-epo.fif')
    avg2 = mne.read_epochs(filename + '_2-epo.fif')

    return avg1, avg2


#----------------------------------------------------------------------------------------------------
def splithalf_and_average(metadata_fields, epochs=None, subj_id=None, data_filenames=None, epoch_filter=None, out_filename=None,
                          decim=1, meg_channels=True, tmin=-0.1, tmax=1.0, baseline=(None, 0), lopass_filter=None, rejection='default',
                          on_response=False, load_error_trials=True):
    """
    Load epochs, split-half them, compute average, and save the averages

    :param on_response: lock epochs on the response
    :type rejection: any
    """

    assert epochs is not None or (data_filenames is not None and subj_id is not None)

    if epochs is None:
        epochs = dpm.files.load_subj_epochs(subj_id, data_filenames, lopass_filter=lopass_filter, decim=decim, meg_channels=meg_channels,
                                            tmin=tmin, tmax=tmax, baseline=baseline, rejection=rejection, on_response=on_response,
                                            load_error_trials=load_error_trials)

    if epoch_filter is not None:
        epochs = epochs[epoch_filter]

    print('\nSplitting epochs into 2 sets...')
    epochs1, epochs2 = umne.split.split_by_event_type(epochs, metadata_fields, round_toward=0)

    print('\nFinished splitting, now computing average epochs...')
    avg1 = umne.epochs.average_epochs_by_metadata(epochs1, metadata_fields)
    avg2 = umne.epochs.average_epochs_by_metadata(epochs2, metadata_fields)

    if out_filename is not None:
        print('\nSaving results to {}_#.fif'.format(out_filename))
        avg1.save(out_filename + '_1-epo.fif')
        avg2.save(out_filename + '_2-epo.fif')

    return avg1, avg2


#=============================================================================
# ======== FUNCTIONS TO CREATE THE PREDICTOR MATRICES FOR REGRESSIONS ========
#=============================================================================


#--------------------------------------------------------------------
def get_digit_per_slot(stimulus):

    pos = stimulus['location']
    digits = dpm.stimuli.as_digits(stimulus['target'])

    slots = [None] * dpm.stimuli.n_positions
    for i in range(len(digits)):
        slots[pos + i] = digits[i]

    return slots


#--------------------------------------------------------------------
def gen_predicted_dissimilarity_2digit(dissimilarity_func):
    """
    Generate a predicted dissimilarity matrix (for all stimuli)
    """
    targets = dpm.stimuli.all_2digit_stimuli()
    targets.sort(key=lambda t: t['target'] * 10 + t['location'])
    md = pd.DataFrame(targets)

    result = umne.rsa.gen_predicted_dissimilarity(dissimilarity_func, md, md)

    return umne.rsa.DissimilarityMatrix([result], md, md)


#=============================================================================
# noinspection PyClassHasNoInit,PyPep8Naming

class dissimilarity:
    """
    Target dissimilarity functions

    Each function gets two dictionnaries containing the fields 'targets' and 'locations' and returns a dissimilarity score (high = dissimilar)
    """

    # ---------------------------------------------------------
    @staticmethod
    def location(stim1, stim2):
        """
        How many digits do not appear in the same locations (the digit itself doensn't matter)
        """

        #-- Array indicating whether each slot contains a digit
        n1 = [d is not None for d in get_digit_per_slot(stim1)]
        n2 = [d is not None for d in get_digit_per_slot(stim2)]

        n_digits_in_same_location = sum([a and b for a, b in zip(n1, n2)])

        return -n_digits_in_same_location


    #---------------------------------------------------------
    class retinotopic_visual_similarity(object):
        """
        Similarity = the visual similarity between digits in the same location
        """
        def __init__(self, similarity_filename):
            self.similarity = dpm.visualsimilarity.load_similarity(similarity_filename, negate=True, identical_digits_default=0)

        def __call__(self, stim1, stim2):
            loc1 = stim1['location']
            loc2 = stim2['location']
            target1 = int(stim1['target'])
            target2 = int(stim2['target'])

            ndig1 = len(str(target1))
            ndig2 = len(str(target2))

            if ndig1 != ndig2:
                return None

            if ndig1 == 2:
                #-- 2-digit numbers

                if abs(loc1 - loc2) != 1:
                    return None

                #-- make sure that stim1 is on the left
                if loc1 > loc2:
                    target1, target2 = target2, target1

                digit1 = target1 // 10  # unit digit
                digit2 = target2 % 10   # decade digit

            else:
                #-- 1-digit numbers
                if loc1 != loc2:
                    return None

                digit1 = target1
                digit2 = target2

            return self.similarity[digit1, digit2]


    #---------------------------------------------------------
    @staticmethod
    def retinotopic_id(stim1, stim2):
        """
        Similarity is the no. of identical digits in an identical location
        Dissimilarity is the same value, negated
        """
        n1 = get_digit_per_slot(stim1)
        n2 = get_digit_per_slot(stim2)
        return - sum([a == b for a, b in zip(n1, n2) if a is not None])


    #---------------------------------------------------------
    @staticmethod
    def decade_unit_id(stim1, stim2):
        return dissimilarity.decade_id(stim1, stim2) + dissimilarity.unit_id(stim1, stim2)


    #---------------------------------------------------------
    @staticmethod
    def decade_id(stim1, stim2):
        """
        1 if the decade digits differ, 0 if the same
        """
        if 'decade' in stim1 and 'decade' in stim2:
            return 0 if stim1['decade'] == stim2['decade'] else 1

        if stim1['target'] < 10 or stim2['target'] < 10:
            raise Exception('dissimilarity.different_decades() cannot be used for 1-digit numbers')

        d1 = int(floor(stim1['target']/10))
        d2 = int(floor(stim2['target']/10))
        return 0 if d1 == d2 else 1


    #---------------------------------------------------------
    @staticmethod
    def decade_group(stim1, stim2):
        """
        Decade ID (2/5 vs. 3/8): this is orthogonal to comparison result
        """
        d1 = int(floor(stim1['target']/10))
        d2 = int(floor(stim2['target']/10))

        assert d1 in (2, 3, 5, 8)
        assert d2 in (2, 3, 5, 8)

        grp1 = d1 in (2, 5)
        grp2 = d2 in (2, 5)
        return 0 if grp1 == grp2 else 1

    #---------------------------------------------------------
    @staticmethod
    def unit_id(stim1, stim2):
        """
        1 if the unit digits differ, 0 if the same
        """
        if 'unit' in stim1 and 'unit' in stim2:
            return 0 if stim1['unit'] == stim2['unit'] else 1

        u1 = stim1['target'] % 10
        u2 = stim2['target'] % 10
        return 0 if u1 == u2 else 1


    #---------------------------------------------------------
    @staticmethod
    def decade_distance(stim1, stim2):
        """
        Numerical distance between the decade digits
        """
        d1 = int(floor(stim1['target']/10))
        d2 = int(floor(stim2['target']/10))
        return abs(d1 - d2)


    #---------------------------------------------------------
    @staticmethod
    def unit_distance(stim1, stim2):
        """
        Numerical distance between the unit digits
        """
        u1 = stim1['target'] % 10
        u2 = stim2['target'] % 10
        return abs(u1 - u2)

    #---------------------------------------------------------
    @staticmethod
    def numerical_distance(stim1, stim2):
        """
        Numerical distance between the unit digits
        """
        return abs(stim1['target'] - stim2['target'])

    #---------------------------------------------------------
    @staticmethod
    class cmp_result(object):
        """
        Whether both targets are small (< reference) or large (> reference) - i.e. yield the same numerical response in the comparison experiment
        """
        def __init__(self, reference=44):
            self._reference = reference

        def __call__(self, stim1, stim2):
            resp1 = stim1['target'] > self._reference
            resp2 = stim2['target'] > self._reference
            return 0 if resp1 == resp2 else 1

    #---------------------------------------------------------
    @staticmethod
    class distance_to_ref(object):
        """
        Targets are similar if their distance to the reference is similar
        """
        def __init__(self, reference=44):
            self._reference = reference

        def __call__(self, stim1, stim2):
            d1 = abs(stim1['target'] - self._reference)
            d2 = abs(stim2['target'] - self._reference)
            return abs(d1 - d2)

        def __str__(self):
            return 'Distance to {}'.format(self._reference)

    #---------------------------------------------------------
    @staticmethod
    def same_digits_any_position(stim1, stim2):
        """
        Whether there is any shared digit whatever the role and the position : example 32 and 25 gives 1
        """
        target1 = stim1['target']
        target2 = stim2['target']

        return sum([a == b for a in str(target1) for b in str(target2) if a is not None])


#--------------------------------------------------------------------
def get_top_triangle_inds(matrix):
    """
    This function is to be used as the "included_cells_getter" parameter of the RSA functions
    """
    md0 = matrix.md0
    md1 = matrix.md1

    def include(i, j):
        return md0['target'][i] > md1['target'][j] or \
               (md0['target'][i] == md1['target'][j] and md0['location'][i] > md1['location'][j])

    return [(i, j) for i in range(len(md0)) for j in range(len(md1)) if include(i, j)]

#--------------------------------------------------------------------
class IncludedCellsDef(object):
    """
    Object to be used as the "included_cells_getter" parameter of the RSA functions.
    This filters some of the matrix cells
    """

    #--------------------------------------------
    def __init__(self, include_digits=None, exclude_digits=None, ndigits=None, only_top_half=False, only_adjacent_locations=False):

        self.include_digits = include_digits
        self.exclude_digits = exclude_digits
        self.ndigits = ndigits
        self.only_top_half = only_top_half
        self.only_adjacent_locations = only_adjacent_locations

    #--------------------------------------------
    def __call__(self, matrix):
        md0 = matrix.md0
        md1 = matrix.md1

        ok_inds_0 = np.where(self.ok_inds(md0))[0]
        ok_inds_1 = np.where(self.ok_inds(md1))[0]

        return [(i, j) for i in ok_inds_0 for j in ok_inds_1 if self.ok_pair(md0, md1, i, j)]

    #--------------------------------------------
    def ok_inds(self, md):
        result = [True] * md.shape[0]

        if self.ndigits is not None:
            result = [res and len(str(target)) == self.ndigits for res, target in zip(result, md['target'])]

        decade = [t//10 for t in md['target']]
        unit = [t%10 for t in md['target']]

        if self.include_digits is not None:
            result = [res and (d in self.include_digits or u in self.include_digits) for res, d, u in zip(result, decade, unit)]

        if self.exclude_digits is not None:
            result = [res and d not in self.exclude_digits and u not in self.exclude_digits for res, d, u in zip(result, decade, unit)]

        return result

    #--------------------------------------------
    def ok_pair(self, md0, md1, i, j):

        if self.only_top_half:
            #-- Require that order(i) > order(j)
            trg_i = md0['target'][i]
            trg_j = md1['target'][j]
            if trg_i < trg_j or (trg_i == trg_j and md0['location'][i] <= md1['location'][j]):
                return False

        if self.only_adjacent_locations and abs(md0['location'][i] - md1['location'][j]) != 1:
            #-- Reqire that i's and j's location would be 1 location apart
            return False

        return True

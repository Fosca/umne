"""
Basic handling of epocsh

Authors: Dror Dotan <dror.dotan@gmail.com>
         Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""

from __future__ import division

import numpy as np
import numbers
import random
import pandas as pd

from autoreject import get_rejection_threshold, compute_thresholds

import mne
import umne


#-------------------------------------------------------------------------
def average_epochs_by_metadata(epochs, metadata_fields):
    """
    Average epochs according to some metadata fields:
    Creating one average epoch for each unique combination of these fields.
    """

    group_id_per_epoch = [tuple(v) for v in epochs.metadata[list(metadata_fields)].values]
    unique_group_ids = set(group_id_per_epoch)

    avg_epochs_data = []
    avg_epochs_metadata = []

    for gid in unique_group_ids:
        curr_group_inds = np.where([i == gid for i in group_id_per_epoch])[0]

        epoch = epochs[curr_group_inds[0]]

        data = epochs[curr_group_inds].get_data().mean(axis=0)
        avg_epochs_data.append(data)

        md = pd.DataFrame({md_field: epoch.metadata[md_field] for md_field in metadata_fields})
        avg_epochs_metadata.append(md)

    return mne.EpochsArray(avg_epochs_data, info=epochs.info, metadata=pd.concat(avg_epochs_metadata), tmin=epochs.tmin)


#------------------------------------------------------------------------------------
def average_epochs(epochs, grouping=None):
    """
    Average epochs by groups: Each set of epochs, marked as one group, are average
    and turn into a single "virtual Epoch".

    In the returned EpochsArray object:

    - *retval.info* is identical with the input data.
    - *retval.events[:, 2]* reflects the input groups (but here, each group number will
      appear just once, indicating the epoch that is the group's average).

    :type epochs: mne.Epochs

    :param grouping: List-like with one entry per epoch, indicating in which groups
        the epoch should be included (and averaged with all epochs of this group).
        Each entry in this list can be either of:

        - An int, indicating a group number in which this epoch will be included.
        - A list of ints, indicating that the epoch should be included in several groups.
        - None, indicating that this epoch should not be included in any group

        If the parameter is not provided, the grouping is taken from epochs.events[:, 2]

    :return: mne.EpochsArray
    """

    if grouping is None:
        #-- Default: use the event specification for grouping
        groups_per_epoch = [(e,) for e in epochs.events[:, 2]]

    elif umne.util.is_collection(grouping):

        if len(grouping) != epochs.events.shape[0]:
            raise Exception('Mismatching arguments: there are {:} epochs but {:} groupings'.format(epochs.events.shape[0], len(grouping)))

        groups_per_epoch = []
        for grp_def in grouping:

            # noinspection PyUnresolvedReferences
            if isinstance(grp_def, int):
                # A single group number
                groups_per_epoch.append((grp_def,))

            elif isinstance(grp_def, (list, tuple, np.ndarray)) and \
                    [isinstance(g, int) for g in grp_def].all():
                # List of group numbers
                groups_per_epoch.append(grp_def)

            elif grp_def is None:
                groups_per_epoch.append([])

            else:
                raise Exception('Invalid entry in the "grouping" argument - expecting int or list[int], got {:}'.format(type(grp_def)))

    else:
        raise Exception('Invalid "grouping" argument - expecting either list of int or list of list[int]')

    data, events = _average_epochs_impl(epochs, groups_per_epoch)

    return mne.EpochsArray(data, epochs.info, events, epochs.tmin)


#------------------------------------------------------------------------------------
def _average_epochs_impl(epochs, groups_per_epoch):
    """
    :type epochs: mne.Epochs
    :param groups_per_epoch: List of lists
    """

    #-- Get raw data as matrix
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape

    #-- Find the list of groups (corresponding with the resulting "virtual epochs")
    uniq_groups = np.unique([g for sublist in groups_per_epoch for g in sublist])
    n_groups = len(uniq_groups)
    if n_groups == 1:
        raise Exception('Only one group was provided')

    #-- Calculate raw data for the result virtual epochs
    result_data = np.zeros([n_groups, n_channels, n_times])
    for i_grp in range(len(uniq_groups)):
        included_epochs = [uniq_groups[i_grp] in g for g in groups_per_epoch]
        if sum(included_epochs) == 1:
            # Only one epoch in this group
            result_data[i_grp, :, :] = data[included_epochs, :, :]
        else:
            # Average several epochs
            result_data[i_grp, :, :] = data[included_epochs, :, :].mean(axis=0)

    #-- Create events
    events = np.zeros([result_data.shape[0], 3], int)
    events[:, 0] = range(1, result_data.shape[0] + 1)
    events[:, 2] = uniq_groups

    return result_data, events


#------------------------------------------------------------------
def subtract_baseline(epochs, baseline_time_range):
    """
    Subtract, from each epoch and channel data, a baseline value.

    The baseline value is the average activation of that channel in a certain time window during that epoch.

    :type epochs: mne.BaseEpochs
    :param baseline_time_range: The time range to use as baseline: (tmin, tmax) pair
    :return: mne.EpochsArray
    """

    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape

    if not umne.util.is_collection(baseline_time_range, allow_set=False) or len(baseline_time_range) != 2:
        raise ValueError("Invalid baseline_time_range ({:}) - expecting a pair of numbers".
                         format(baseline_time_range, epochs.tmin, epochs.tmax))

    for t in baseline_time_range:
        if not isinstance(t, numbers.Number) or not (epochs.tmin <= t <= epochs.tmax):
            raise ValueError("Invalid baseline_time_range value ({:}) - expecting a number between tmin({:}) and tmax({:})".
                             format(t, epochs.tmin, epochs.tmax))

    if baseline_time_range[0] > baseline_time_range[1]:
        baseline_time_range = baseline_time_range[1], baseline_time_range[0]

    #-- Get matrix indices of baseline values
    baseline_inds = [baseline_time_range[0] <= t <= baseline_time_range[1] for t in epochs.times]
    if len(baseline_inds) == 0:
        raise ValueError("Invalid baseline_time_range value ({:}) - No data point matches this time window".format(baseline_time_range))

    #-- Remove baseline from each channel in each epoch
    for i_epoch in range(n_epochs):
        for i_channel in range(n_channels):
            baseline = data[i_epoch, i_channel, baseline_inds].mean()
            data[i_epoch, i_channel, :] = data[i_epoch, i_channel, :] - baseline

    return mne.EpochsArray(data, epochs.info, epochs.events, epochs.tmin)


#------------------------------------------------------------------
def create_epochs_from_raw(raw, events, metadata=None, meg_channels=True, tmin=-0.1, tmax=0.4, decim=10, reject=None, baseline=(None, 0)):
    """
    Create epochs for decoding

    :param raw:
    :type raw: mne.io.BaseRaw
    :param reject: Either of:
                'auto_global': Automatically compute rejection threshold based on all data
                'auto_channel': Automatically compute rejection threshold for each channel
                'default': Use default values
                None: no rejection
                A dict with the entries 'mag'/'grad'/both: set these rejection parameters (if mag/grad unspecified: no rejection for these channels)

    :param events: The definition of epochs and their event IDs (#epochs x 3 matrix)
    """

    events = np.array(events)

    picks_meg = mne.pick_types(raw.info, meg=meg_channels, eeg=False, eog=False, stim=False, exclude='bads')

    if reject == 'auto_global':
        epochs = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, proj=True, picks=picks_meg, baseline=baseline)
        ep_reject = get_rejection_threshold(epochs, decim=2)

    elif reject == 'auto_channel':
        print('Auto-detecting rejection thresholds per channel...')
        epochs = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, proj=True, picks=picks_meg, baseline=baseline)
        ep_reject = compute_thresholds(epochs, picks=picks_meg, method='random_search', augment=False, verbose='progressbar')

    else:
        ep_reject = _get_rejection_thresholds(reject, meg_channels)

    epochs = mne.Epochs(raw, events=events, metadata=metadata, tmin=tmin, tmax=tmax, proj=True, decim=decim,
                        picks=picks_meg, reject=ep_reject, preload=True, baseline=baseline)

    # print("\nEvenr IDs:")
    # for cond, eid in epochs.event_id.items():
    #     print("Condition '%s' (event_id = %d): %d events" % (cond, eid, len(epochs[cond])))

    return epochs


#----------------------------------------
def _get_rejection_thresholds(reject, meg_channels):

    if reject is None:
        return None

    result = dict()

    if reject == 'default':
        #-- Apply defaults
        if meg_channels == 'mag' or meg_channels is True:
            result['mag'] = 4e-12

        if meg_channels == 'grad' or meg_channels is True:
            result['grad'] = 4000e-13

    else:
        #-- Specific definition provided
        if meg_channels is True:
            return reject

        if meg_channels == 'mag' and 'mag' in reject:
            result['mag'] = reject['mag']

        elif meg_channels == 'grad':
            result['grad'] = reject['grad']

    return result


#------------------------------------------------------------------
def recode_groups(epochs, event_to_group_func, group_name_id=None, in_place=False,
                  equalize_event_counts=True):
    """
    Assign epochs to new groups (by assigning a new event ID to each epoch).
    Returns an Epochs object with updated events.

    :param epochs: Epochs to update
    :type epochs: mne.BaseEpochs

    :param event_to_group_func: A function that gets an event ID of an epoch and returns a
                       new event ID.

    :param group_name_id: Group IDs to consider for analysis in the new epochs. In this dict,
                        the key is a string name and the value is an event ID
    :type group_name_id: dict

    :param in_place: Whether to modify the input "epochs" object (True) or create a copy (False).
    :type in_place: bool

    :return: mne.BaseEpochs
    """

    if not in_place:
        epochs = epochs.copy()

    curr_groups = epochs.events[:, 2]
    new_groups = [event_to_group_func(e) for e in curr_groups]
    epochs.events[:, 2] = new_groups

    if group_name_id is not None:
        epochs.event_id = group_name_id.copy()

    if equalize_event_counts:
        eid = list(np.unique(epochs.events[:, 2]) if group_name_id is None else group_name_id.keys())
        epochs.equalize_event_counts(eid)

    return epochs


#-------------------------------------------------------------------
def choose_random_epochs(triggers, n_epochs_per_trigger=1):

    triggers = np.array(triggers)
    use_epoch = np.array([False] * len(triggers))

    for trigger in np.unique(triggers):
        inds = np.where(triggers == trigger)[0]
        random.shuffle(inds)
        use_inds = inds[1:min(n_epochs_per_trigger + 1, len(inds))]
        use_epoch[use_inds] = True

    return use_epoch

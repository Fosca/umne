"""
Functions for splitting data

Authors: Dror Dotan <dror.dotan@gmail.com>
         Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""

import random
import numpy as np
import math


#--------------------------------------------------------------------------------------------
# todo: we might need to add "condition1, condition2" parameters (or something similar) to define how to split trials to 2 sets

def split_by_event_type(epochs, metadata_fields, set1_proportion=0.5, round_toward=0):
    """
    Get epochs, divide them into groups, and split the epochs of each group into 2 sets

    Return two Epochs objects

    :param epochs: input epochs
    :param metadata_fields: List of metadata field names. Epochs are divided into groups according to these fields.
    :param set1_proportion: The size of resulting epochs set #1
    :param round_toward: In case an epoch group cannot be divided exactly into 2 according to set1_proportion, this determines how to
                         round the set sizes: 1 = make set#1 larger; 2 = make set#2 larger; 0 (default) = randomize for each group, according to
                         the proportion
    """

    assert 0 < set1_proportion < 1, "Invalid set1_proportion ({:}), expecting a value between 0 and 1".format(set1_proportion)
    assert round_toward in (0, 1, 2), "Invalid round_toward ({:}) - expecting 1, 2, or 0 for random".format(round_toward)

    group_id_per_epoch = [tuple(v) for v in epochs.metadata[list(metadata_fields)].values]

    set_per_epoch = _split_ids_into_groups(group_id_per_epoch, set1_proportion, round_toward)

    return epochs[set_per_epoch == 1], epochs[set_per_epoch == 2]


#-------------------------------------------------------------------------
def _split_ids_into_groups(group_id_per_epoch, set1_proportion, round_toward):

    result = np.ones(len(group_id_per_epoch)) * 1


    for event_id in set(group_id_per_epoch):

        #-- Get the indices of epochs with event_id, in random order
        in_curr_group = [id == event_id for id in group_id_per_epoch]
        curr_group_epoch_inds = np.where(in_curr_group)[0]
        random.shuffle(curr_group_epoch_inds)
        curr_group_epoch_inds = np.array(curr_group_epoch_inds)
        curr_group_size = len(curr_group_epoch_inds)

        #-- Select which of the group's epochs goes to result set #1

        if round_toward == 0:
            rnd = 1 if random.random() < ((curr_group_size * set1_proportion) % 1) else 2
        else:
            rnd = round_toward
        round_func = math.floor if rnd == 1 else math.ceil

        max_ind = int(round_func(curr_group_size * (1 - set1_proportion)))

        group_2_inds = curr_group_epoch_inds[list(range(max_ind))]
        result[group_2_inds] = 2

    return result

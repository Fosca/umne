"""
Transformers of data matrices of epochs or epoch-like

Authors: Dror Dotan <dror.dotan@gmail.com>
         Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""

import numpy as np
import random

from sklearn.base import TransformerMixin


#=========================================================================================================

class ZScoreEachChannel(TransformerMixin):
    """
    Z-score the data of each channel separately

    Input matrix: Epochs x Channels x TimePoints
    Output matrix: Epochs x Channels x TimePoints (same size as input)
    """

    #--------------------------------------------------
    def __init__(self, debug=False):
        self._debug = debug

    #--------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    #--------------------------------------------------
    def transform(self, x):
        result = np.zeros(x.shape)
        n_epochs, nchannels, ntimes = x.shape
        for c in range(nchannels):
            channel_data = x[:, c, :]
            m = np.mean(channel_data)
            sd = np.std(channel_data)
            if self._debug:
                print('ZScoreEachChannel: channel {:} m={:}, sd={:}'.format(c, m, sd))
            result[:, c, :] = (x[:, c, :]-m)/sd

        return result


#=========================================================================================================

class SlidingWindow(TransformerMixin):
    """
    Aggregate time points in a "sliding window" manner

    Input: Anything x Anything x Time points
    Output - if averaging: Unchanged x Unchanged x Windows
    Output - if not averaging: Windows x Unchanged x Unchanged x Window size
                Note that in this case, the output may not be a real matrix in case the last sliding window is smaller than the others
    """

    #--------------------------------------------------
    def __init__(self, window_size, step, min_window_size=None, average=True, debug=False):
        """
        :param window_size: The no. of time points to average
        :param step: The no. of time points to slide the window to get the next result
        :param min_window_size: The minimal number of time points acceptable in the last step of the sliding window.
                                If None: min_window_size will be the same as window_size
        :param average: If True, just reduce the number of time points by averaging over each window
                        If False, each window is copied as-is to the output, without averaging
        """
        self._window_size = window_size
        self._step = step
        self._min_window_size = min_window_size
        self._average = average
        self._debug = debug


    #--------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    #--------------------------------------------------
    def start_window_inds(self, n_time_points):
        min_window_size = self._min_window_size or self._window_size
        return np.array(range(0, n_time_points-min_window_size+1, self._step))

    #--------------------------------------------------
    def transform(self, x):
        x = np.array(x)
        assert len(x.shape) == 3
        n1, n2, n_time_points = x.shape

        #-- Get the start-end indices of each window
        window_start = self.start_window_inds(n_time_points)
        if len(window_start) == 0:
            #-- There are fewer than window_size time points
            raise Exception('There are only {:} time points, but at least {:} are required for the sliding window'.
                            format(n_time_points, self._min_window_size))
        window_end = window_start + self._window_size
        window_end[-1] = min(window_end[-1], n_time_points)  # make sure that the last window doesn't exceed the input size

        if self._debug:
            win_info = [(s, e, e-s) for s, e in zip(window_start, window_end)]
            print('SlidingWindow transformer: the start,end,length of each sliding window: {:}'.
                  format(win_info))
            if len(win_info) > 1 and win_info[0][2] != win_info[-1][2] and not self._average:
                print('SlidingWindow transformer: note that the last sliding window is smaller than the previous ones, ' +
                      'so the result will be a list of 3-dimensional matrices, with the last list element having ' +
                      'a different dimension than the previous elements. ' +
                      'This format is acceptable by the RiemannDissimilarity transformer')

        if self._average:
            #-- Average the data in each sliding window
            result = np.zeros((n1, n2, len(window_start)))
            for i in range(len(window_start)):
                result[:, :, i] = np.mean(x[:, :, window_start[i]:window_end[i]], axis=2)

        else:
            #-- Don't average the data in each sliding window - just copy it
            result = []
            for i in range(len(window_start)):
                result.append(x[:, :, window_start[i]:window_end[i]])

        return result


#=========================================================================================================

class AveragePerEvent(TransformerMixin):
    """
    This transformer averages all epochs that have the same label.
    It can also create several averages per event ID (based on independent sets of trials)

    Input matrix: Epochs x Channels x TimePoints
    Output matrix: Labels x Channels x TimePoints. If asked to create N results per event ID, the "labels"
                   dimension is multiplied accordingly.
    """

    #--------------------------------------------------
    def __init__(self, event_ids=None, n_results_per_event=1, max_events_with_missing_epochs=0, debug=False):
        """
        :param event_ids: The event IDs to average on. If None, compute average for all available events.
        :param n_results_per_event: The number of aggregated stimuli to create per event type.
               The event's epochs are distributed randomly into N groups, and each group is averaged, creating
               N independent results.
        :param max_events_with_missing_epochs: The maximal number of event IDs for which we allow the number
               of epochs to be smaller than 'n_results_per_event'. For such events, randomly-selected epochs
               will be duplicated.
        """
        assert isinstance(n_results_per_event, int) and n_results_per_event > 0
        assert isinstance(max_events_with_missing_epochs, int) and max_events_with_missing_epochs >= 0

        self._event_ids = None if event_ids is None else np.array(event_ids)
        self._curr_event_ids = None
        self._n_results_per_event = n_results_per_event
        self._max_events_with_missing_epochs = max_events_with_missing_epochs
        self._debug = debug

        if debug:
            if event_ids is None:
                print('AveragePerEvent: will create averages for all events.')
            else:
                print('AveragePerEvent: will create averages for these events: {:}'.format(event_ids))


    #--------------------------------------------------
    # noinspection PyUnusedLocal,PyAttributeOutsideInit
    def fit(self, x, y, *_):

        self._y = np.array(y)

        if self._event_ids is None:
            self._curr_event_ids = np.unique(y)
            if self._debug:
                print('AveragePerEvent: events IDs are {:}'.format(self._event_ids))
        else:
            self._curr_event_ids = self._event_ids

        return self


    #--------------------------------------------------
    def transform(self, x):

        x = np.array(x)

        result = []

        #-- Split the epochs by event ID.
        #-- x_per_event_id has a 3-dim matrix for each event ID
        x_per_event_id = [x[self._y == eid] for eid in self._curr_event_ids]

        #-- Check if there are enough epochs per event ID
        too_few_epochs = [len(e) < self._n_results_per_event for e in x_per_event_id]  # list of bool - one per event ID
        if sum(too_few_epochs) > self._max_events_with_missing_epochs:
            raise Exception('There are {:} event IDs with fewer than {:} epochs: {:}'.
                            format(sum(too_few_epochs), self._n_results_per_event,
                            self._curr_event_ids[np.where(too_few_epochs)[0]]))
        elif sum(too_few_epochs) > 0:
            print('WARNING (AveragePerEvent): There are {:} event IDs with fewer than {:} epochs: {:}'.
                  format(sum(too_few_epochs), self._n_results_per_event,
                         self._curr_event_ids[np.where(too_few_epochs)[0]]))

        #-- Do the actual aggregation
        for i in range(len(x_per_event_id)):
            # Get a list whose length is n_results_per_event; each list entry is a 3-dim matrix to average
            agg = self._aggregate(x_per_event_id[i])
            if self._debug:
                print('AveragePerEvent: event={:}, #epochs={:}'.format(self._curr_event_ids[i], [len(a) for a in agg]))
            result.extend([np.mean(a, axis=0) for a in agg])

        result = np.array(result)

        if self._debug:
            print('AveragePerEvent: transformed from shape={:} to shape={:}'.format(x.shape, result.shape))

        return result

    #--------------------------------------------------
    def _aggregate(self, one_event_x):
        """
        Distribute the epochs of one_event_x into separate sets

        The function returns a list with self._n_results_per_event different sets.
        """

        if self._n_results_per_event == 1:
            #-- Aggregate all epochs into one result
            return [one_event_x]

        if len(one_event_x) >= self._n_results_per_event:

            #-- The number of epochs is sufficient to have at least one different epoch per result

            one_event_x = np.array(one_event_x)

            result = [[]] * self._n_results_per_event

            #-- First, distribute an equal number of epochs to each result
            n_in_epochs = len(one_event_x)
            in_epochs_inds = range(len(one_event_x))
            random.shuffle(in_epochs_inds)
            n_take_per_result = int(np.floor(n_in_epochs / self._n_results_per_event))
            for i in range(self._n_results_per_event):
                result[i] = list(one_event_x[in_epochs_inds[:n_take_per_result]])
                in_epochs_inds = in_epochs_inds[n_take_per_result:]

            #-- If some epochs remained, add each of them to a different result set
            n_remained = len(in_epochs_inds)
            for i in range(n_remained):
                result[i].append(one_event_x[in_epochs_inds[i]])

        else:

            #-- The number of epochs is too small: each result will consist of a single epoch, and epochs some will be duplicated

            #-- First, take all events that we have
            result = list(one_event_x)

            #-- Then pick random some epochs and duplicate them
            n_missing = self._n_results_per_event - len(result)
            epoch_inds = range(len(one_event_x))
            random.shuffle(epoch_inds)
            duplicated_inds = epoch_inds[:n_missing]
            result.extend(np.array(result)[duplicated_inds])

            result = [[x] for x in result]

        random.shuffle(result)

        return result

"""
Infra for Representational Similarity Analysis

Authors: Dror Dotan <dror.dotan@gmail.com>
         Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""
import numpy as np
import pickle
import pandas as pd
import glob
import math
import multiprocessing
import os
import re

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import spearmanr
import scipy.interpolate
from mne.decoding import UnsupervisedSpatialFilter
from mne.utils import ProgressBar
from statsmodels import api as sm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages

from umne import transformers
import umne


plt.rcParams['animation.ffmpeg_path'] = u'/usr/local/bin/ffmpeg'

default_colors = 'black', 'r', 'g', 'b', 'orange', 'purple', 'grey'


#----------------------------------------------------------------------------------------
class DissimilarityMatrix(object):
    """
    A series of dissimilarity matrices - one per time point

    In matrix.data, the first index is the time points

    NOTE that the matrix axes are not necessarily sorted, and that rows and columns may have a different order.
    """

    #-----------------------------------------
    def __init__(self, data, md_ax0, md_ax1, times=None, epochs0_info=None, epochs1_info=None):

        self.data = np.array(data)
        assert self.data.ndim == 3, "Invalid no. of dimensions for input data ({:})".format(self.data.ndim)
        assert self.data.shape[1] == len(md_ax0)
        assert self.data.shape[2] == len(md_ax1)

        self.md0 = md_ax0.reset_index(drop=True)
        self.md1 = md_ax1.reset_index(drop=True)

        self.times = times

        self.epochs0_info = epochs0_info
        self.epochs1_info = epochs1_info


    #-----------------------------------------
    def get_axis_labels(self, axis, md_fields, field_value_format="{0}={1}", separator="; "):
        """
        Get the labels of one of the matrid axes

        :param axis: 1 or 2
        :param md_fields: field names to use for labels
        :param field_value_format: A string for str.format(), which determines the label of one field ({0} is the field name, {1} is its value)
        :param separator: Separator between the different MD fields
        """
        if axis == 0:
            field_values = np.array(self.md0[md_fields])
        elif axis == 1:
            field_values = np.array(self.md1[md_fields])
        else:
            raise Exception('Invalid axis {:}'.format(axis))

        labels = []
        for i in range(field_values.shape[0]):
            tmp = separator.join([field_value_format.format(fld, value) for fld, value in zip(md_fields, field_values[i, :])])
            labels.append(tmp)

        return labels

    #-----------------------------------------
    def __str__(self):
        return "DissimilarityMatrix[{:} x {:}]".format(len(self.md0), len(self.md1))


    #-----------------------------------------
    def save(self, filename, mkdir=True):
        """
         Save the DissimilarityMatrix object to a file

         :param mkdir: Create the directory if it does not exist
         """
        if mkdir:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)


    #-----------------------------------------
    @staticmethod
    def load(filename):
        """
        Load a DissimilarityMatrix object that was saved using DissimilarityMatrix.save()
        """
        with open(filename, 'rb') as fp:
            dissim = pickle.load(fp)

        assert isinstance(dissim, DissimilarityMatrix), \
            "Invalid data in {:}: expecting a {:} object".format(filename, DissimilarityMatrix.__name__)

        return dissim


    #----------------------------------------------------------------------------------------------
    @property
    def size(self):
        """Get the matrix size - excluding the time-point dimension"""
        return tuple(self.data.shape[1:])


    #----------------------------------------------------------------------------------------------
    @property
    def is_square(self):
        return self.data.shape[1] == self.data.shape[2]


    #----------------------------------------------------------------------------------------------
    @property
    def n_timepoints(self):
        return self.data.shape[0]


    #----------------------------------------------------------------------------------------------
    def copy(self):
        """Create a copy of the matrix"""
        new_data = np.array(self.data).copy()
        return DissimilarityMatrix(new_data, self.md0, self.md1, self.epochs0_info, self.epochs1_info)


    #----------------------------------------------------------------------------------------------
    def zscore(self, inplace=False):
        """
        Normalize a dissimilarity matrix by z-scoring it

        :return: The normalized matrix
        """

        assert self.is_square, "Matrix is not square (size = {:})".format(self.size)

        matrix = self if inplace else self.copy()

        for i in range(matrix.n_timepoints):
            zscore_matrix(matrix[i, :i, :], inplace=True)

        return matrix


    #----------------------------------------------------------------------------------------------
    def filter(self, filter_text, ax0=False, ax1=False, inplace=False):
        """
        Filter rows/columns of the matrix

        :param filter_text: The filter to apply (pandas query on metadata)
        :type filter_text: str
        :param ax0: Whether to apply the filter to axis=0
        :param ax1: Whether to apply the filter to axis=1
        :param inplace: If true, modify 'self'
        """

        assert ax0 or ax1, "Specify at least one filtering axis to DissimilarityMatrix.filter()"

        matrix = self if inplace else self.copy()

        if ax0:
            inds0 = list(self.md0.query(filter_text).index)
            matrix.md0 = matrix.md0.loc[inds0]
        else:
            inds0 = list(range(self.size[0]))

        if ax1:
            inds1 = list(self.md1.query(filter_text).index)
            matrix.md1 = matrix.md1.loc[inds1]
        else:
            inds1 = list(range(self.size[1]))

        matrix.data = np.array([t[inds0, :][:, inds1] for t in matrix.data])

        return matrix


    #----------------------------------------------------------------------------------------------
    def reorder(self, md_fields0, md_fields1=None):
        return reorder_matrix(self, md_fields0, md_fields1)


    #----------------------------------------------------------------------------------------------
    def diagonal_inds(self, unique_md_fields=None):
        return _get_diagonal_inds(self, unique_md_fields)


    #----------------------------------------------------------------------------------------------
    def diagonal(self):
        return [self.data[i, j] for i, j in self.diagonal_inds()]


    #----------------------------------------------------------------------------------------------
    def sort(self, key):
        """
        Sort the matrix and the stimuli in a given order

        Return (matrix, stimuli) - both as np.array

        :param matrix: an N*N dissimilarity matrix
        :param stimuli: a list of N stimuli (the axis of the matrix)
        :param stim_compare_function: a function that compares 2 stimuli (compatible with Python's sorted())
        """

        #-- 1. convert md0 into a list of dictionaties, and same for md1
        md0 = self.md0.to_dict()
        md1 = self.md1.to_dict()
        for i, m in enumerate(md0):
            m['_pre_sort_index_'] = i
        for i, m in enumerate(md1):
            m['_pre_sort_index_'] = i

        #-- 2. Sort
        md0.sort(key=key)
        md1.sort(key=key)

        #-- 3. Reorgaize the matrix according to md0._pre_sort_index_ and md1._pre_sort_index_
        self.md0 = pd.DataFrame(md0)
        self.md1 = pd.DataFrame(md1)
        self.data = self.data[[m['_pre_sort_index_'] for m in md0], [m['_pre_sort_index_'] for m in md1]]

        return self


#----------------------------------------------------------------------------------------------
def _get_diagonal_inds(matrix, unique_md_fields=None):

    md0 = matrix.md0
    md1 = matrix.md1

    if unique_md_fields is None:
        unique_md_fields = set(md0.keys())
        assert unique_md_fields == set(md1.keys()), \
            "When matrix rows and columns have different keys, you must provide the keys in order to compute the diagonal"

    def is_diagonal(i, j):
        return sum([md0[k][i] != md1[k][j] for k in unique_md_fields]) == 0

    return [(i, j) for i in range(len(md0)) for j in range(len(md1)) if is_diagonal(i, j)]


#----------------------------------------------------------------------------------------------
def gen_observed_dissimilarity(epochs0, epochs1, n_pca=30, metric='spearmanr',
                               sliding_window_size=None, sliding_window_step=None, sliding_window_min_size=None,
                               debug=None):
    """
    Generate the observed dissimilarity matrix

    :param epochs0: Epohcs, averaged over the relevant parameters
    :param epochs1: Epohcs, averaged over the relevant parameters
    :param n_pca: the number of PCA components.
    :param metric: The metric to use when calculating distance between instances in a feature array, for
            non-Riemannian dissimilarity.
            If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist
            for its metric parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
            If metric is precomputed, X is assumed to be a distance matrix.
            Alternatively, if metric is a callable function, it is called on each pair of instances (rows)
            and the resulting value recorded.
            The callable should take two arrays from X as input and return a value indicating the distance between them.
    :param sliding_window_size: If specified (!= None), the data will be averaged using a sliding window before
                    computing dissimilarity. This parameter is the number of time points included in each window
    :param sliding_window_step: The number of time points for sliding the window on each step
    :param sliding_window_min_size: The minimal number of time points acceptable in the last step of the sliding window.
                                If None: min_window_size will be the same as window_size

    :return: np.array
    """

    #-- Validate input
    assert (sliding_window_size is None) == (sliding_window_step is None), \
        "Either sliding_window_size and sliding_window_step are both None, or they are both not None"
    debug = debug or set()

    if metric == 'mahalanobis' and n_pca is not None:
        print('WARNING: PCA should not be used for metric=mahalanobis, ignoring this parameter')
        n_pca = None

    #-- Original data: #epochs x Channels x TimePoints
    data1 = epochs0.get_data()
    data2 = epochs1.get_data()

    #-- z-scoring doesn't change the data dimensions
    data1 = transformers.ZScoreEachChannel(debug=debug is True or 'zscore' in debug).fit_transform(data1)
    data2 = transformers.ZScoreEachChannel(debug=debug is True or 'zscore' in debug).fit_transform(data2)

    #-- Run PCA. Resulting data: Epochs x PCA-Components x TimePoints
    if n_pca is not None:
        pca = UnsupervisedSpatialFilter(PCA(n_pca), average=False)
        combined_data = np.vstack([data1, data2])
        pca.fit(combined_data)
        data1 = pca.transform(data1)
        data2 = pca.transform(data2)

    #-- Apply a sliding window
    #-- Result in non-Riemann mode: epochs x Channels/components x TimePoints
    #-- todo: Result in Riemann: TimeWindows x Stimuli x Channels/components x TimePoints-within-one-window; will require SlidingWindow(average=False)
    if sliding_window_size is None:
        times = epochs0.times

    else:
        xformer = transformers.SlidingWindow(window_size=sliding_window_size, step=sliding_window_step,
                                             min_window_size=sliding_window_min_size)
        data1 = xformer.fit_transform(data1)
        data2 = xformer.fit_transform(data2)

        mid_window_inds = xformer.start_window_inds(len(epochs0.times)) + round(sliding_window_size/2)
        times = epochs0.times[mid_window_inds]

    #-- Get the dissimilarity matrix
    #-- Result: Time point x epochs1 x epochs2
    dissim_matrices = _compute_dissimilarity(data1, data2, metric, debug is True or 'dissim' in debug)
    # todo in Riemann: xformer = RiemannDissimilarity(metric=riemann_metric, debug=debug is True or 'dissim' in debug)

    assert len(dissim_matrices) == len(times), "There are {} dissimilarity matrices but {} times".format(len(dissim_matrices), len(times))

    return DissimilarityMatrix(dissim_matrices, epochs0.metadata, epochs1.metadata, times=times,
                               epochs0_info=epochs0.info, epochs1_info=epochs1.info)


#----------------------------------------------------------------------------------------
def _compute_dissimilarity(data1, data2, metric, debug=False):

    if metric == 'spearmanr':
        metric = _spearmanr

    assert data1.shape[1] == data1.shape[1], "Expecting the same number of channels"
    assert data1.shape[2] == data1.shape[2], "Expecting the same number of time points"

    n_timepoints = data1.shape[2]

    if debug:
        print('Computing dissimilarity (method={}): computing a {}*{} dissimilarity matrix using correlations for each of {} time points...'.
              format(metric, data1.shape[0], data2.shape[0], n_timepoints))

    pb = ProgressBar(n_timepoints, mesg="Computing dissimilarity")

    def run_per_timepoint(t):
        d = pairwise_distances(data1[:, :, t], data2[:, :, t], metric=metric, n_jobs=multiprocessing.cpu_count())
        pb.update(t + 1)
        return d

    dissim_matrix_per_timepoint = np.asarray([run_per_timepoint(t) for t in range(n_timepoints)])

    return dissim_matrix_per_timepoint


def _spearmanr(a, b):
    # noinspection PyUnresolvedReferences
    return 1 - spearmanr(a, b).correlation


#----------------------------------------------------------------------------------------
def reorder_matrix(matrix, md_fields0, md_fields1=None):
    """
    Reorder the axes of the dissimilarity matrix. The matrix is not changed - a new matrix is returned.

    :param matrix: The matrix to reorder
    :param md_fields0: List of metadata fields that determine the order of axis 0 in the new matrix
    :param md_fields1: List of metadata fields that determine the order of axis 1 in the new matrix
    """

    if md_fields1 is None:
        md_fields1 = md_fields0

    new_md0 = matrix.md0.sort_values(md_fields0)
    sort_inds0 = list(new_md0.index)
    new_md0 = new_md0.reset_index(drop=True)

    new_md1 = matrix.md1.sort_values(md_fields1)
    sort_inds1 = list(new_md1.index)
    new_md1 = new_md1.reset_index(drop=True)

    new_data = np.zeros(matrix.data.shape)
    for i0 in range(len(sort_inds0)):
        for i1 in range(len(sort_inds1)):
            new_data[:, i0, i1] = matrix.data[:, sort_inds0[i0], sort_inds1[i1]]

    return DissimilarityMatrix(new_data, new_md0, new_md1, matrix.epochs0_info, matrix.epochs1_info)


#----------------------------------------------------------------------------------------------
def average_matrices(matrices, averaging_method='linear'):
    """
    Get a list of dissimilarity matrices and average them.
    All matrices are assumed to be of the same size and with the same metadata and epochs_info

    Return a DissimilarityMatrix object with 1 time point

    :param matrices: An array of dissimilarity matrices (all must have the same size)
    :param averaging_method: linear - standard average
                             square - square each value, then average, then sqrt
    """

    m = matrices[0]
    data = np.array([d.data for d in matrices])

    if averaging_method == 'linear':

        average_data = np.mean(data, axis=0)

    elif averaging_method == 'square':

        n1, n2 = m.size
        average_data = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                values = np.array(data[:, i, j])
                sq_mean = np.sqrt((values ** 2).mean())
                average_data[i, j] = sq_mean * np.sign(values.mean())

    else:
        raise Exception('Invalid averaging_method({:})'.format(averaging_method))

    return DissimilarityMatrix(average_data, m.md0, m.md1, m.epochs0_info, m.epochs1_info)


#----------------------------------------------------------------------------------------------
def aggregate_matrix(matrix, md_fields0, md_fields1=None):
    """
    Aggregate some rows/columns of the matrix. The new matrix will contain fewer rows & columns; each row/column
    in the new matrix is the average of several rows/columns in the original matrix

    :param md_fields0: List of metadata fields to use for grouping rows (group rows with identical values)
    :param md_fields1: List of metadata fields to use for grouping columns (group rows with identical values)
    """

    md_fields1 = md_fields1 or md_fields0

    #-- Compute the metadata of the new matrix: unique values of the relevant fields
    new_md0 = matrix.md0[md_fields0].drop_duplicates()
    new_md1 = matrix.md1[md_fields1].drop_duplicates()
    new_md0.reset_index(drop=True, inplace=True)
    new_md1.reset_index(drop=True, inplace=True)
    n0 = len(new_md0)
    n1 = len(new_md1)

    #-- Get the relevant metadata fields for each row/column of the existing matrix, as a tuple
    ax0_groups = [tuple(v) for v in matrix.md0[list(md_fields0)].values]
    ax1_groups = [tuple(v) for v in matrix.md1[list(md_fields1)].values]

    #-- Get the relevant metadata fields for each row/column of the new matrix, as a tuple
    new_ax0_groups = [tuple(v) for v in new_md0.values]
    new_ax1_groups = [tuple(v) for v in new_md1.values]

    #-- Average cells
    data = np.zeros((matrix.n_timepoints, n0, n1))
    for i in range(n0):
        inds0 = [g == new_ax0_groups[i] for g in ax0_groups]
        for j in range(n1):
            inds1 = [g == new_ax1_groups[j] for g in ax1_groups]
            for t in range(data.shape[0]):
                data[t, i, j] = matrix.data[t, inds0, :][:, inds1].mean()

    return DissimilarityMatrix(data, new_md0, new_md1, matrix.epochs0_info, matrix.epochs1_info)


#----------------------------------------------------------------------------------------------
def concatenate(matrix1, matrix2, axis=0):
    """
    Concatenate two dissimilarity matrices

    :param axis: The axis along which to merge (the other axis must be identical in the 2 matrices)
    """
    assert axis in (0, 1), "Invalid axis ({:})".format(axis)

    other_axis = 1 - axis
    assert matrix1.size[other_axis] == matrix2.size[other_axis], \
        "Matrix sizes mismatch (size1={:}, size2={:}, axis={:}}".format(matrix1.size, matrix2.size, axis)

    if axis == 0:
        new_md0 = pd.concat([matrix1.md0, matrix1.md0])
        new_md1 = matrix1.md1

    else:
        new_md0 = matrix1.md0
        new_md1 = pd.concat([matrix1.md1, matrix1.md1])

    newdata = np.concatenate([matrix1.data, matrix2.data], axis=axis)

    return DissimilarityMatrix(newdata, new_md0, new_md1)  # todo epochs info?


#----------------------------------------------------------------------------------------------
def zscore_matrix(matrix, inplace=False):
    """
    Normalize a dissimilarity matrix by z-scoring it

    :param matrix: a 2-dimensioned np.array
    :return: The normalized matrix
    """

    matrix = matrix if inplace else matrix.copy()

    matrix = (matrix - np.mean(matrix)) / np.std(matrix)

    return matrix


#================================================================================================
#                      GENERATE THE PREDICTOR MATRICES
#================================================================================================

#----------------------------------------------------------------------------------------------
def gen_predicted_dissimilarity(dissimilarity_func, md0, md1, ensure_diagonal_is_0=True, zscore=False):
    """
    Generate a matrix with the dissimilarity matrix according to the chosen dissimilarity function
    and the metadata fields of the observed dissimilarity md0 and md1.

    :param dissimilarity_func: A function that gets 2 targets and returns their dissimilarity
                       (available functions: in class 'dissimilarity' below)
    :param md0, md1: Metadata fields of the observed dissimilarity object
    """

    n0 = len(md0)
    n1 = len(md1)
    result = np.zeros([n0, n1])

    diagonal_ind = None

    for i0 in range(n0):
        for i1 in range(n1):
            result[i0, i1] = dissimilarity_func(md0.iloc[i0], md1.iloc[i1])
            if diagonal_ind is None and dict(md0.iloc[i0]) == dict(md1.iloc[i1]):  # this point is on the diagonal
                diagonal_ind = i0, i1

    if zscore:
        result = zscore_matrix(result)

    if ensure_diagonal_is_0:
        assert diagonal_ind is not None
        result -= result[diagonal_ind[0], diagonal_ind[1]]

    return result


#================================================================================================
#                    REGRESSION ANALYSIS MATRICES
#================================================================================================

#----------------------------------------------------------------------------------------------
def regress_dissimilarity(observed_dissimilarity, gen_predictor_funcs, zscore=True, excluded_cells_getter=None, included_cells_getter=None):
    """
    Regress the dissimilarity matrix in a multiple regression against several predicted matrices

    Return an array-of-arrays = TimePoint x Predictor matrix of coefficients

    :param observed_dissimilarity: The matrix generated from the MEG data
    :param gen_predictor_funcs: A list of functions, each will generate a dissimilarity matrix
                       (to be used as regression predictor)
    :param excluded_cells_getter: Function that gets DissimilarityMatrix, returns list of indices (x,y) of cells to be excluded
    :param included_cells_getter: Function that gets DissimilarityMatrix, returns list of indices (x,y) of cells to be included
    """

    md0 = observed_dissimilarity.md0
    md1 = observed_dissimilarity.md1

    assert excluded_cells_getter is None or included_cells_getter is None

    if excluded_cells_getter is not None:
        if excluded_cells_getter == 'diagonal':
            excluded_cells_getter = _get_diagonal_inds
        include_cells = np.full((len(md0), len(md1)), True)
        for i, j in excluded_cells_getter(observed_dissimilarity):
            include_cells[i, j] = False

    elif included_cells_getter is not None:
        include_cells = np.full((len(md0), len(md1)), False)
        for i, j in included_cells_getter(observed_dissimilarity):
            include_cells[i, j] = True

    else:
        include_cells = None

    if include_cells is None:
        print('Regressing a full dissimilarity matrix')
    else:
        include_cells = np.reshape(include_cells, [include_cells.size])
        assert sum(include_cells) > 0, "ERROR: No cells were included"
        print('Regressing {:}/{:} cells of the dissimilarity matrix'.format(sum(include_cells), len(include_cells)))

    predicted_dissimilarity = []
    for i, predfunc in enumerate(gen_predictor_funcs):
        # todo: I've set ensure_diagonal_is_0=False because otherwise, for some predictor it yields NAN values. However, this may be a problem if we didn't split-half
        predval = gen_predicted_dissimilarity(predfunc, md0, md1, ensure_diagonal_is_0=False)
        pred_array = _matrix_to_array(predval, include_cells)
        if sum([math.isnan(x) for x in pred_array]) > 0:
            raise Exception('Predictor #{} yielded some undefined similarity values'.format(i))
        if len(set(pred_array)) == 1:
            raise Exception('Predictor #{} value is constant ({})'.format(i, pred_array[0]))
        predicted_dissimilarity.append(pred_array)

    predicted_dissimilarity = np.transpose(np.array(predicted_dissimilarity))

    if zscore:
        #-- We z-score after filtering include_cells, so that the excluded cells (whose value is invalid) will not affect the z scoring
        _zscore_each_predictor(predicted_dissimilarity)

    result = []
    for i in range(observed_dissimilarity.n_timepoints):
        rr = _regress_dissimilarity_one_tp(observed_dissimilarity.data[i, :, :], predicted_dissimilarity, include_cells,
                                           zscore_observed=zscore)
        result.append(rr)

    return result


#----------------------------------------------------------
def _matrix_to_array(m, include_cells):

    m = np.reshape(m, [m.size])

    if include_cells is not None:
        m = m[include_cells]

    return m


#----------------------------------------------------------
def _zscore_each_predictor(predictors):
    for i in range(predictors.shape[1]):
        predictors[:, i] = (predictors[:, i] - predictors[:, i].mean()) / predictors[:, i].std()


#----------------------------------------------------------------------------------------------
def _regress_dissimilarity_one_tp(observed_dissimilarity, predicted_dissimilarity, include_cells, zscore_observed):
    """
    Regress and return the regression coefficients. The last returned coefficient is const.
    """

    observed_dissimilarity = _matrix_to_array(observed_dissimilarity, include_cells)
    if zscore_observed:
        observed_dissimilarity = (observed_dissimilarity - np.average(observed_dissimilarity)) / np.std(observed_dissimilarity)

    #-- add constant to predictors
    predicted_dissimilarity = np.column_stack((predicted_dissimilarity, np.ones(len(predicted_dissimilarity))))

    # lin_reg = LinearRegression()
    # lin_reg.fit(predicted_dissimilarity, observed_dissimilarity)
    results = sm.OLS(observed_dissimilarity, predicted_dissimilarity).fit()

    return results.params


#------------------------------------------------------------------------------------
def load_and_regress_dissimilarity(filename_mask, gen_predictor_funcs, filename_subj_id_pattern='.*_(\\w+)_lock.*.dmat', cell_filter=None,
                                   zscore_predictors=True, included_cells_getter=None, excluded_cells_getter=None):
    """
    Regress the dissimilarity matrix according to several predictors

    return a subjects x timepoints x predictors matrix of regression coefficients

    :param gen_predictor_funcs: A list of functions, each can generate a predictor
    :param cell_filter: Filter for dissimilarity row/column
    :type cell_filter: str
    """

    path_str = os.path.normpath(filename_mask).split(os.sep)
    print('\nRegressing dissimilarity for .../{} with the following predictors:\n{}\n'.
          format(os.sep.join(path_str[-2:]), ", ".join([_genpred_func_name(f) for f in gen_predictor_funcs])))

    all_dissim_filenames = glob.glob(filename_mask)
    matching_filenames = [f for f in all_dissim_filenames if re.match(filename_subj_id_pattern, os.path.basename(f))]

    print('Regressing {}/{} dissimilarity matrices:\n{}\n'.format(len(matching_filenames), len(all_dissim_filenames), "\n".join(matching_filenames)))
    if len(all_dissim_filenames) > len(matching_filenames):
        print('WARNING: Note that some dissimilarity-matrix files were excluded')

    times = None

    regression_all_participants = []
    subj_ids = []
    for i_file, dissim_filename in enumerate(matching_filenames):
        m = re.match(filename_subj_id_pattern, os.path.basename(dissim_filename))
        subj_ids.append(m.group(1))

        with open(dissim_filename, "rb") as fid:
            d = pickle.load(fid)
        dissim_1subj = DissimilarityMatrix(d.data, md_ax0=d.md0, md_ax1=d.md1, epochs0_info=d.epochs0_info, epochs1_info=d.epochs1_info, times=d.times)

        if cell_filter is not None:
            dissim_1subj = dissim_1subj.filter(cell_filter, ax0=True, ax1=True)

        print('File {}/{}: '.format(i_file, len(matching_filenames)), end='')
        regression_1subj = regress_dissimilarity(dissim_1subj, gen_predictor_funcs, zscore=zscore_predictors,
                                                 included_cells_getter=included_cells_getter, excluded_cells_getter=excluded_cells_getter)
        regression_1subj = [x.tolist() for x in regression_1subj]
        regression_all_participants.append(regression_1subj)

        if times is None and hasattr(dissim_1subj, 'times'):
            times = dissim_1subj.times

    return np.array(regression_all_participants), subj_ids, times


#-----------------------
def _genpred_func_name(gen_predictor_func):

    fdir = dir(gen_predictor_func)

    if '__name__' in fdir:
        return gen_predictor_func.__name__

    if '__str__' in fdir:
        return str(gen_predictor_func)

    if '__class__' in fdir:
        return gen_predictor_func.__class__.__name__

    return '?'


#------------------------------------------------------------------------------------
def plot_regression_results_per_sujb(regression_results, subj_ids, times, save_as,
                                     line_width=0.5, figure_id=1, colors=default_colors, alpha=0.15,
                                     pred_mult_factor=None, legend=None, cols_per_page=2, rows_per_page=4):
    """
    Plot the RSA regression results

    :param regression_results: #Subjects x #TimePoints x #Predictors matrix
    :param times: x axis values
    :param figure_id:
    :param colors: Array, one color for each predictor
    :param alpha: For the shaded part that represents 1 standard error
    :param pred_mult_factor: Multiple each predictor by this factor (array, size = # of predictors)
    :param subject_x_factor: The x values of each subject are stretched by this factor.
    :param legend: Legend names (array, size = # of predcitors)
    :param reset: If True (default), the figure is cleared before plotting
    """
    n_subj, n_tp, n_pred = regression_results.shape
    assert n_subj == len(subj_ids)

    regression_results = _normalize_rr(regression_results, n_pred, pred_mult_factor, None)

    if legend is not None:
        assert len(legend) == n_pred, "There are {} legend entries but {} predictors".format(len(legend), n_pred)

    n_trials_per_page = cols_per_page * rows_per_page

    pdf = PdfPages(save_as)

    plt.close(figure_id)

    subj_inds = list(range(n_subj))
    n_done = 0

    while len(subj_inds) > 0:

        curr_page_n_plots = min(n_trials_per_page, len(subj_inds))

        fig, axes = plt.subplots(rows_per_page, cols_per_page)
        fig.subplots_adjust(hspace=.8, wspace=0.3)

        axes = np.reshape(axes, [n_trials_per_page])

        for i in range(curr_page_n_plots):
            subj_ind = subj_inds.pop()
            n_done += 1
            ax = axes[i]
            ax.set_title('Subject {}'.format(subj_ids[subj_ind]), fontdict=dict(fontsize=5))
            _plot_rr_1subj(ax, regression_results[subj_ind, :, :], times, colors, legend=None, linewidth=line_width)

        if curr_page_n_plots < n_trials_per_page:
            for i in range(curr_page_n_plots, n_trials_per_page):
                ax = axes[i]
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)

        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()


#------------------------------------------------------------------------------------
def _plot_rr_1subj(ax, subj_rr, times, colors, linewidth=None, legend=None):

    _, n_pred = subj_rr.shape

    ax.plot(times, [0] * len(times), color='black', label='_nolegend_', linewidth=linewidth)

    for ipred in range(n_pred):
        curr_predictor_rr = subj_rr[:, ipred]
        ax.plot(times, curr_predictor_rr, color=colors[ipred], linewidth=linewidth)

    if legend is not None:
        ax.legend(legend, loc="upper left")


#------------------------------------------------------------------------------------
def plot_regression_results(regression_results, times,
                            show_significance=False, significance_time_window=None, p_threshold=0.05, line_width_significant=2, line_width_ns=0.5,
                            figure_id=1, colors=default_colors, alpha=0.15,
                            pred_mult_factor=None, subject_x_factor=None, legend=None, reset=True, save_as=None):
    """
    Plot the RSA regression results

    :param regression_results: #Subjects x #TimePoints x #Predictors matrix
    :param times: x axis values
    :param figure_id:
    :param colors: Array, one color for each predictor
    :param alpha: For the shaded part that represents 1 standard error
    :param pred_mult_factor: Multiple each predictor by this factor (array, size = # of predictors)
    :param subject_x_factor: The x values of each subject are stretched by this factor.
    :param legend: Legend names (array, size = # of predcitors)
    :param reset: If True (default), the figure is cleared before plotting
    """
    n_subj, n_tp, n_pred = regression_results.shape

    regression_results = _normalize_rr(regression_results, n_pred, pred_mult_factor, subject_x_factor)

    if legend is not None:
        assert len(legend) == n_pred, "There are {} legend entries but {} predictors".format(len(legend), n_pred)

    if reset:
        plt.close(figure_id)

    fig = plt.figure(figure_id)
    ax = plt.gca()

    plt.plot(times, [0] * len(times), color='black', label='_nolegend_')

    for ipred in range(n_pred):
        curr_predictor_rr = regression_results[:, :, ipred]
        coeffs = np.mean(curr_predictor_rr, axis=0)
        coeffs_se = np.std(curr_predictor_rr, axis=0) / math.sqrt(n_subj)

        #-- plot
        plt.fill_between(times, coeffs - coeffs_se, coeffs + coeffs_se, color=colors[ipred], alpha=alpha, label='_nolegend_')

        if show_significance:
            line_widths = np.array([line_width_ns] * len(times))
            significant = _pred_coeff_significant(times, curr_predictor_rr, significance_time_window, p_threshold)
            line_widths[significant] = line_width_significant

            points = [(t, c) for t, c in zip(times, coeffs)]
            segments = [(p1, p2) for p1, p2 in zip(points[:-1], points[1:])]  # each 2 adjacent points are a segment
            lc = LineCollection(segments, linewidths=line_widths[:-1], color=colors[ipred])
            ax.add_collection(lc)

        else:
            plt.plot(times, coeffs, color=colors[ipred])

    if legend is not None:
        plt.legend(legend, loc="upper left")

    if save_as is not None:
        fig = plt.gcf()
        fig.savefig(save_as)

    return fig


#----------------------------------------------------------
def _normalize_rr(regression_results, n_pred, pred_mult_factor, subject_x_factor):
    """
    Normalize regression results: multiply by a factor, and stretch each subject along the x axis

    :param regression_results:
    :param n_pred:
    :param pred_mult_factor:
    :param subject_x_factor:
    """
    if pred_mult_factor is not None:
        assert len(pred_mult_factor) == n_pred
        regression_results = regression_results.copy()
        for ipred in range(n_pred):
            regression_results[:, :, ipred] *= pred_mult_factor[ipred]

    if subject_x_factor is not None:
        regression_results = _stretch_rr_on_t_axis(regression_results, subject_x_factor)

    return regression_results


#----------------------------------------------------------
def _pred_coeff_significant(times, curr_predictor_rr, time_window, p_threshold):
    significant = np.array([False] * len(times))
    if time_window is None:
        time_window_inds = [True] * len(times)
    else:
        time_window_inds = np.logical_and(times >= time_window[0], times <= time_window[1])
    pval_per_time = umne.stats.stats_cluster_based_permutation_test(curr_predictor_rr[:, time_window_inds])
    significant[time_window_inds] = pval_per_time <= p_threshold
    return significant


#----------------------------------------------------------
def _stretch_rr_on_t_axis(regression_results, subject_x_factor):
    """
    "Stretch" each subject's regression results on the time axis according to a per-subject factor (which presumably reflects
    each subject's relative processing speed).

    The total number of time points does not change. If a subject data is "condensed", the last time point will be duplicated to complete
    the required number of time pointss
    """

    n_subj, n_tp, n_pred = regression_results.shape

    assert len(subject_x_factor) == n_subj

    subject_x_factor = np.array(subject_x_factor) / np.mean(subject_x_factor)
    print(list(subject_x_factor))

    new_rr = np.zeros([n_subj, n_tp, n_pred])

    timepoints = np.array(range(n_tp))

    #-- Interpolate regression results
    for i_subj in range(n_subj):
        for ipred in range(n_pred):

            if subject_x_factor[i_subj] < 1:
                index_of_1 = int(math.floor(n_tp * subject_x_factor[i_subj]))
                new_rr[i_subj, index_of_1:, ipred] = regression_results[i_subj, -1, ipred]
                max_interp = index_of_1
            else:
                max_interp = n_tp

            cs = scipy.interpolate.interp1d(timepoints * subject_x_factor[i_subj], regression_results[i_subj, :, ipred])
            new_rr[i_subj, :max_interp, ipred] = cs(timepoints[:max_interp])

            '''
            if ipred == 0:
                print('=============== factor={}, From'.format(subject_x_factor[i_subj]))
                print(regression_results[i_subj, :, ipred])
                print('=============== To')
                print(new_rr[i_subj, :, ipred])
            '''

    return new_rr


#---------------------------------------------------------
def _get_md_label(md):
    return " ".join(['{:}={:}'.format(k, md[k]) for k in md.keys()])


def extract_ticks_labels_from_md(metadata, get_label=_get_md_label):
    xticks_labels = [get_label(row) for ind, row in metadata.iterrows()]
    return np.array(xticks_labels)


#---------------------------------------------------------
# noinspection PyUnresolvedReferences
def plot_dissimilarity(dissim, max_value=None, colormap=cm.viridis, tick_filter=None, get_label=_get_md_label):
    """
    Plot a dissimilarity matrix

    :type dissim: DissimilarityMatrix
    :param max_value: Maximal value in the plot
    :param tick_filter: A function that gets a metadata and returns True/False - whether to use it as a tick
    :param get_label: A function that gets the list of ticks from the full metadata of one matrix axis
    """

    matrix = dissim.data

    min_val = 0
    if max_value is None:
        max_val = np.mean(matrix) + np.std(matrix)
    else:
        max_val = max_value

    plt.imshow(np.mean(matrix, axis=0), interpolation='none', cmap=colormap, origin='upper', vmin=min_val, vmax=max_val)
    plt.colorbar()
    x_ticks = extract_ticks_labels_from_md(dissim.md0, get_label)
    y_ticks = extract_ticks_labels_from_md(dissim.md1, get_label)
    y = np.array(range(matrix.shape[1]))
    x = np.array(range(matrix.shape[2]))

    if tick_filter is not None:
        include_x = [tick_filter(t) for i, t in dissim.md0.iterrows()]
        include_y = [tick_filter(t) for i, t in dissim.md1.iterrows()]

        x_ticks = x_ticks[include_x]
        x = x[include_x]
        y_ticks = y_ticks[include_y]
        y = y[include_y]

    plt.xticks(x, x_ticks, rotation='vertical')
    plt.yticks(y, y_ticks)


#---------------------------------------------------------
def extract_ticks_labels_from_md1(metadata):

    xticks_labels = []
    for m in range(len(metadata)):
        string_fields = ''
        for field in metadata.keys():
            string_fields += '%s_%s_' % (field[:3], str(metadata[field][m]))

        xticks_labels.append(string_fields)

    return np.array(xticks_labels)


#---------------------------------------------------------
# noinspection PyUnresolvedReferences
def video_dissimilarity_matrices(dissimilarity, save_path_video, tmin=-500, tmax=700, interval=100, vmin=None, vmax=None, cmap=cm.viridis):

    data_dissimilarity = dissimilarity.data
    n_times, n_comps, n_comps = data_dissimilarity.shape

    time_stamps = np.round(np.linspace(tmin, tmax, n_times), 3)

    if vmin is None:
        vmin = np.mean(data_dissimilarity) - np.std(data_dissimilarity)
    if vmax is None:
        max_val = np.mean(data_dissimilarity) + np.std(data_dissimilarity)
    else:
        max_val = vmax

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes()
    ttl = ax.text(.5, 1.05, '', transform=ax.transAxes, va='center')

    im = plt.imshow(data_dissimilarity[0, :, :], vmin=vmin, vmax=max_val, interpolation='none', animated=True, cmap=cmap)
    plt.colorbar()
    x_ticks = extract_ticks_labels_from_md(dissimilarity.md0)
    y_ticks = extract_ticks_labels_from_md(dissimilarity.md1)
    y = range(data_dissimilarity.shape[1])
    x = range(data_dissimilarity.shape[2])
    plt.xticks(x, x_ticks, rotation='vertical', fontsize=12)
    plt.yticks(y, y_ticks, fontsize=12)

    def init():
        ttl.set_text('')
        return ttl

    def updatefig(k):
        im.set_array(np.transpose(data_dissimilarity[k % n_times, :, :]))
        ttl.set_text('%i ms' % time_stamps[k])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, init_func=init, frames=n_times-1, interval=interval, blit=False)
    plt.show()

    ani.save(save_path_video)

    return True

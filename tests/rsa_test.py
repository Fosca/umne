import unittest
from src.umne.rsa import DissimilarityMatrix
import numpy as np

def mock_dissimilarity_matrix(shape,md0,md1):
    """
    The first dimension is supposed to be time and the other two ones
    the dimension for which the dissimilarity is computed.
    """

    data = np.zeros(shape)
    return DissimilarityMatrix(data, md0, md1)


class DissimilarityMatrixTest(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, False)

    def square_is_square(self):
        dissim = mock_dissimilarity_matrix([10,2,2], ['md0_0','md0_1'], ['md1_0','md1_1'])
        self.assertEqual(True, dissim.is_square())

    def rectange_is_not_square(self):
        dissim = mock_dissimilarity_matrix([10,1,2], ['md0_0'], ['md1_0','md1_1'])
        self.assertEqual(False, dissim.is_square())

    def strange_size_not_dissim(self):

        dissim = mock_dissimilarity_matrix([10,2,2,4], ['md0_0','md0_1'], ['md1_0','md1_1'])
        # when generating the dissimilarity matrix, in the __init__, there will be an assertion that will stop the
        # building of the object
        self.assertEqual(AssertionError,"Invalid no. of dimensions for input data ({:})".format(dissim.data.ndim))

    def check_reordering(self):
        dissim = mock_dissimilarity_matrix([10,2,2], ['md0_0','md0_1'], ['md1_0','md1_1'])
        dissim.reorder(['md0_1','md0_0'],['md1_1','md1_0'])
        self.assertEqual(dissim.md0, ['md0_1','md0_0'])
        self.assertEqual(dissim.md1, ['md1_1','md1_0'])

    def impossible_reordering(self):
        # maybe it would be good to make sure the metadata fields are all different ?
        dissim = mock_dissimilarity_matrix([10,2,2], ['md0_0','md0_1'], ['md1_0','md1_1'])
        dissim.reorder(['md0_0','md0_0'],['md1_1','md1_0'])
        # see what the pandas resorting function would output if it is impossible

    def filtering(self):
        dissim = mock_dissimilarity_matrix([10,2,2], ['md0_0','md0_1'], ['md1_0','md1_1'])
        dissim.filter('md0_0',ax0=True)
        self.assertEqual(dissim.md0, ['md0_0'])

    def filtering_impossible(self):
        # check what the dataframe.query outputs when it fails

    def compute_diagonal_on_rectangle(self):
        dissim = mock_dissimilarity_matrix([10,2,3], ['md0_0','md0_1'], ['md1_0','md1_1','md1_2'])
        self.assertEqual(AssertionError,"When matrix rows and columns have different keys, you must provide the keys in order to compute the diagonal")

    def compute_diagonal_on_square_with_different_fields(self):
        dissim = mock_dissimilarity_matrix([10,2,2], ['md0_0','md0_1'], ['md1_0','md1_1'])
        self.assertEqual(AssertionError,"When matrix rows and columns have different keys, you must provide the keys in order to compute the diagonal")

    def compute_diagonal(self):
        dissim = mock_dissimilarity_matrix([10,2,2], ['md0_0','md0_1'], ['md0_0','md0_1'])
        diag = dissim.diagonal()
        self.assertEqual(2,len(diag))

if __name__ == '__main__':
    unittest.main()

# Filename: cider.py
#
#
# Description: Describes the class to compute the CIDEr
# (Consensus-Based Image Description Evaluation) Metric
#          by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and
# Tsung-Yi Lin <tl483@cornell.edu>

from .cider_scorer import CiderScorer


class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, n=4, df="corpus"):
        """
        Initialize the CIDEr scoring function
        : param n (int): n-gram size
        : param df (string): specifies where to get the IDF values from
                    takes values 'corpus', 'coco-train'
        : return: None
        """
        # set cider to sum over 1 to 4-grams
        self._n = n
        self._df = df

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        : param  gts (dict) : {image:tokenized reference sentence}
        : param res (dict)  : {image:tokenized candidate sentence}
        : return: cider (float) : computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n)

        for image_id, hypo in res.items():
            ref = gts[image_id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score(self._df)

        return score, scores

    def method(self):
        return "CIDEr"

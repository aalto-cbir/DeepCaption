import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from utils import to_contiguous
from vocabulary import caption_ids_to_words

import sys

#sys.path.append("cider")
#from pyciderevalcap.ciderD.ciderD import CiderD

#sys.path.append("coco-caption")
#from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None


# CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(greedy_sample, sample, target, scorers, vocab, image_ids):#(greedy_res, gen_result, data, opt):
    batch_size = sample.size(0)  # batch_size = sample_size * seq_per_img
    #seq_per_img = batch_size // len(data['gts'])

    scorer = scorers['CIDEr']
    gts = {}
    res = {}
    res_greedy = {}

    for j in range(target.shape[0]):
        jid = image_ids[j]
        if jid not in gts:
            gts[jid] = []
        # if params.hierarchical_model:
        #     gts[jid].append(paragraph_ids_to_words(captions[j], vocab).lower())
        # else:
        gts[jid].append(caption_ids_to_words(target[j], vocab).lower())

    for j in range(sample.shape[0]):
        jid = image_ids[j]
        # if params.hierarchical_model:
        #     res[jid] = [paragraph_ids_to_words(sampled_ids_batch[j], vocab).lower()]
        # else:
        res[jid] = [caption_ids_to_words(sample[j], vocab).lower()]

    for j in range(greedy_sample.shape[0]):
        jid = image_ids[j]
        # if params.hierarchical_model:
        #     res_greedy[jid] = [paragraph_ids_to_words(sampled_ids_batch[j], vocab).lower()]
        # else:
        res_greedy[jid] = [caption_ids_to_words(greedy_sample[j], vocab).lower()]

    _, score = scorer.compute_score(gts, res)
    _, score_baseline = scorer.compute_score(gts, res_greedy)

    scores = score - score_baseline

    rewards = np.repeat(scores[:, np.newaxis], sample.size(1), 1)

    return rewards

    # ----------------------------------------

    # res = OrderedDict()
    #
    # gen_result = gen_result.data.cpu().numpy()
    # greedy_res = greedy_res.data.cpu().numpy()
    # for i in range(batch_size):
    #     res[i] = [array_to_str(gen_result[i])]
    # for i in range(batch_size):
    #     res[batch_size + i] = [array_to_str(greedy_res[i])]
    #
    # gts = OrderedDict()
    # for i in range(len(data['gts'])):
    #     gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
    #
    # res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    # res__ = {i: res[i] for i in range(2 * batch_size)}
    # gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    # if opt.cider_reward_weight > 0:
    #     _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    #     print('Cider scores:', _)
    # else:
    #     cider_scores = 0
    # if opt.bleu_reward_weight > 0:
    #     _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
    #     bleu_scores = np.array(bleu_scores[3])
    #     print('Bleu scores:', _[3])
    # else:
    #     bleu_scores = 0
    # scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
    #
    # scores = scores[:batch_size] - scores[batch_size:]
    #
    # rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    #
    # return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, sequence_logprobs, reward):
        # Mask tokens out if they're forced. I.e. when start of sentence token is always given instead of predicted.
        # Can be done with something like mask = (sample > 0).float() (would not work here bc our tokens are positive)
        #sequence_logprobs = to_contiguous(sequence_logprobs).view(-1)
        #reward = to_contiguous(reward).view(-1)
        #mask = (sequence > 0).float()  # select all words, but tokens like eos (would not work here bc our tokens are positive)
        # set the first entry at 1
        #mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        if reward.dtype != sequence_logprobs.dtype:
            reward = reward.type(sequence_logprobs.type())

        #output = - sequence_logprobs * reward * mask
        #output = torch.sum(output) / torch.sum(mask) #word-wise instead of caption-wise. This is inherited from original neuraltalk2

        return torch.mean(- sequence_logprobs * reward)

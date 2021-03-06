import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from collections import Counter
from vocabulary import word_ids_to_words, clean_word_ids

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SharedEmbeddingXentropyLoss(nn.Module):
    def __init__(self, param_lambda=0.15):
        super(SharedEmbeddingXentropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.param_lambda = param_lambda

    def forward(self, P, outputs, targets):
        """:param P - projection matrix of size (hidden_size x hidden_size)"""
        reg_term = self.param_lambda * P.norm()

        return self.loss(outputs, targets) + reg_term


class HierarchicalXEntropyLoss(nn.Module):
    def __init__(self, weight_sentence_loss=5.0, weight_word_loss=1.0):
        super(HierarchicalXEntropyLoss, self).__init__()
        self.weight_sent = torch.Tensor([weight_sentence_loss]).to(device)
        self.weight_word = torch.Tensor([weight_word_loss]).to(device)
        self.sent_loss = nn.CrossEntropyLoss()
        self.word_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """ outputs and targets are both tuples of length = 2
        First element: a Tensor of size (batchsize X max_sentences)
            it tells us the sentence RNN stopping indices
        Second element:  a list of length=batchsize, of Tensors of
            size at most = max_sentence the actual captions for each sentence position"""
        # Compare number of sentences?
        # outputs: tuple = (captions, stopping_states)
        # targets: target paragraphs
        # Do regular loss on individual sentences?
        # 1) Cross entropy loss of current sentence being last sentence
        # 2) Cross entropy loss of curent word being correct
        # For each sample in a mini-batch we want the number of sentences
        # Max number of sentences in the target mini-batch:
        max_sentences = len(targets[1])
        # print('max_sentences: {}'.format(max_sentences))

        self.loss_s = torch.Tensor([0]).to(device)

        for j in range(max_sentences):
            # print("Size of outputs[0][{}]: {}, size of targets[0][{}] {}".
            #      format(j, outputs[0][j].size(), j, targets[0][j].size()))
            self.loss_s += self.sent_loss(outputs[0][:, j], targets[0][:, j])

        self.loss_w = torch.Tensor([0]).to(device)

        for j in range(max_sentences):
            # print("Size of outputs[1][{}]: {}, size of targets[1][{}] {}".
            #      format(j, outputs[1][j].size(), j, targets[1][j].size()))
            self.loss_w += self.word_loss(outputs[1][j], targets[1][j])

        return self.weight_sent * self.loss_s + self.weight_word * self.loss_w

    def item_terms(self):
        return self.weight_sent, self.loss_s, self.weight_word, self.loss_w


def get_caps(greedy_sample, sample, vocab, skip_start_token=False, keep_tokens=False):
    res_sample = word_ids_to_words(sample, vocab, skip_start_token=skip_start_token, keep_tokens=keep_tokens)
    res_greedy = word_ids_to_words(greedy_sample, vocab, skip_start_token=skip_start_token, keep_tokens=keep_tokens)

    return res_sample, res_greedy


def get_self_critical_reward(res_sample, res_greedy, gts, scorers, max_len=20):
    batch_size = len(res_sample)
    scorer = scorers['CIDEr-D']

    gts = dict(enumerate(gts + gts))
    res = [{'image_id': i, 'caption': r} for i, r in enumerate(list(res_sample.values()) + list(res_greedy.values()))]

    _, scores = scorer.compute_score(gts, res)
    scores = scores[:batch_size] - scores[batch_size:]

    reward = np.repeat(scores[:, np.newaxis], max_len, 1)
    reward = torch.from_numpy(reward).float().to(device)
    return reward, scores.mean()


def create_mask(sample, vocab):
    """
    Mask tokens out if they're forced. I.e. when start of sentence token is always given instead of predicted,
    so no loss would be needed for it.
    Mask also tokens after <end>, because they shouldn't get any gradient.
    :param sample:
    :param vocab:
    :return:
    """
    sample_max_len = sample.size(1)
    mask = torch.zeros(sample.size()).float().to(device)
    end_idx = [(sample[i] == vocab('<end>')).nonzero() for i in range(sample.size(0))]  # because no dim=0 parameter
    end_idx = [i[0].item() + 1 if i.size(0) != 0 else sample_max_len for i in end_idx]  # + 1 because end is included

    for i, ei in enumerate(end_idx):
        mask[i][:ei] = 1

    return mask


class SelfCriticalLoss(nn.Module):
    """Reinforcement Learning Self-critical loss.
    https://arxiv.org/abs/1612.00563
    Code from https://github.com/ruotianluo/self-critical.pytorch
    """
    def __init__(self):
        super(SelfCriticalLoss, self).__init__()

    def forward(self, sample, sample_log_probs, greedy_sample, gts_batch, scorers, vocab, return_advantage=False):
        assert sample.shape[0] == sample_log_probs.shape[0] == greedy_sample.shape[0] == len(gts_batch)

        # Don't look for starting token because is always first given and then discarded, during sampling
        res_sample, res_greedy = get_caps(greedy_sample, sample, vocab, skip_start_token=True)
        reward, advantage = get_self_critical_reward(res_sample, res_greedy, gts_batch, scorers, max_len=sample.size(1))

        # Mask tokens out after <end>, and tokens not generated by the network
        mask = create_mask(sample, vocab)

        loss = - (sample_log_probs * reward * mask).sum() / mask.sum()

        return (loss, advantage) if return_advantage else loss


class SelfCriticalWithDiversityLoss(nn.Module):
    def __init__(self):
        super(SelfCriticalWithDiversityLoss, self).__init__()

    def forward(self, sample, sample_log_probs, greedy_sample, gts_batch, scorers, vocab, return_advantage=False):
        assert sample.shape[0] == sample_log_probs.shape[0] == greedy_sample.shape[0] == len(gts_batch)

        # Don't look for starting token because is always first given and then discarded, during sampling
        res_sample, res_greedy = get_caps(greedy_sample, sample, vocab, skip_start_token=True)
        reward, advantage = get_self_critical_reward(res_sample, res_greedy, gts_batch, scorers, max_len=sample.size(1))

        # Mask tokens out after <end>, and tokens not generated by the network
        mask = create_mask(sample, vocab)

        accuracy = - (sample_log_probs * reward * mask).sum() / mask.sum()
        diversity = self.diversity(res_sample)

        # gamma = 0.5
        # loss = gamma * accuracy + (1 - gamma) * diversity
        loss = accuracy + diversity
        return (loss, advantage) if return_advantage else loss

    @staticmethod
    def diversity(res):
        """
        Diversity loss is taken and adapted for 3 and 4-gram from:
        FACE: Improving Neural Response Diversity with Frequency-Aware Cross-Entropy Loss
        https://arxiv.org/abs/1902.09191.pdf
        Code from https://github.com/ShaojieJiang/FACE
        :param res:
        :return:
        """
        unigram, bigram, trigram, cuatrigram = set(), set(), set(), set()
        n_tokens = 0

        for a, b in res.items():
            cap = b[0].split()
            v_len = len(cap)
            n_tokens += v_len
            unigram.update(cap)
            bigram.update([tuple(cap[i:i + 2]) for i in range(len(cap) - 1)])
            trigram.update([tuple(cap[i:i + 3]) for i in range(len(cap) - 2)])
            cuatrigram.update([tuple(cap[i:i + 4]) for i in range(len(cap) - 3)])

        d1 = len(unigram) / n_tokens
        d2 = len(bigram) / n_tokens
        d3 = len(trigram) / n_tokens
        d4 = len(cuatrigram) / n_tokens

        return - (d1 + d2 + d3 + d4)


class SelfCriticalWithRelativeDiversityLoss(nn.Module):
    def __init__(self):
        super(SelfCriticalWithRelativeDiversityLoss, self).__init__()

    def forward(self, sample, sample_log_probs, greedy_sample, gts_batch, scorers, vocab, return_advantage=False):
        assert sample.shape[0] == sample_log_probs.shape[0] == greedy_sample.shape[0] == len(gts_batch)

        # Don't look for starting token because is always first given and then discarded, during sampling
        res_sample, res_greedy = get_caps(greedy_sample, sample, vocab, skip_start_token=True)
        reward, advantage = get_self_critical_reward(res_sample, res_greedy, gts_batch, scorers, max_len=sample.size(1))

        # Mask tokens out after <end>, and tokens not generated by the network
        mask = create_mask(sample, vocab)

        accuracy = - (sample_log_probs * reward * mask).sum(dim=1) / mask.sum(dim=1)
        diversity, _ = self.relative_diversity(res_sample, gts_batch)
        diversity = torch.from_numpy(diversity).type_as(accuracy)

        # gamma = 0.5
        # loss = gamma * accuracy + (1 - gamma) * diversity
        loss = accuracy + diversity
        loss = loss.mean()
        return (loss, advantage) if return_advantage else loss

    def relative_diversity(self, res, gts, dist='l1'):
        gts_ngrams = self.ngram_counter(gts)
        res_ngrams = self.ngram_counter(res.values())
        ret = np.zeros(len(res))

        for i in range(len(res)):
            gts_exclusives = gts_ngrams[i].keys() - res_ngrams[i].keys()
            res_exclusives = res_ngrams[i].keys() - gts_ngrams[i].keys()
            common = set(list(res_ngrams[i].keys()) + list(gts_ngrams[i].keys())) - gts_exclusives - res_exclusives

            gts_vals = np.array([gts_ngrams[i][k] for k in gts_exclusives])
            res_vals = np.array([res_ngrams[i][k] for k in res_exclusives])
            common_vals = np.array([gts_ngrams[i][k] for k in common]) - np.array([res_ngrams[i][k] for k in common])

            if dist == 'l1':
                gts_vals = gts_vals.sum()
                res_vals = res_vals.sum()
                common_vals = np.abs(common_vals).sum()
            elif dist == 'l2':
                gts_vals = np.power(gts_vals, 2).sum()
                res_vals = np.power(res_vals, 2).sum()
                common_vals = np.power(np.abs(common_vals), 2).sum()
            else:
                raise ValueError('Bad dist')

            ret[i] = common_vals + gts_vals + res_vals

        ret = ret * 0.1
        return ret, ret.mean()

    @staticmethod
    def ngram_counter(caps):
        ngram_list = []
        ngram_counter = Counter()

        for b in caps:
            for c in b:
                cap = c.split()
                ngram_counter.update(cap)
                ngram_counter.update([tuple(cap[i:i + 2]) for i in range(len(cap) - 1)])
                ngram_counter.update([tuple(cap[i:i + 3]) for i in range(len(cap) - 2)])
                ngram_counter.update([tuple(cap[i:i + 4]) for i in range(len(cap) - 3)])

            i_ngrams = {}
            for k, v in ngram_counter.items():
                i_ngrams[k] = v / len(b)

            ngram_counter.clear()
            ngram_list.append(i_ngrams)

        return ngram_list


class SelfCriticalWithBLEUDiversityLoss(nn.Module):
    def __init__(self):
        super(SelfCriticalWithBLEUDiversityLoss, self).__init__()

    def forward(self, sample, sample_log_probs, greedy_sample, gts_batch, scorers, vocab, return_advantage=False):
        assert sample.shape[0] == sample_log_probs.shape[0] == greedy_sample.shape[0] == len(gts_batch)

        # Don't look for starting token because is always first given and then discarded, during sampling
        res_sample, res_greedy = get_caps(greedy_sample, sample, vocab, skip_start_token=True)
        reward, advantage = get_self_critical_reward(res_sample, res_greedy, gts_batch, scorers, max_len=sample.size(1))

        # Mask tokens out after <end>, and tokens not generated by the network
        mask = create_mask(sample, vocab)

        accuracy = - (sample_log_probs * reward * mask).sum(dim=1) / mask.sum(dim=1)
        diversity = self.diversity(res_sample, gts_batch)
        diversity = torch.from_numpy(diversity).type_as(accuracy)

        # gamma = 0.5
        # loss = gamma * accuracy + (1 - gamma) * diversity
        loss = accuracy + diversity * 0.15
        loss = loss.mean()
        return (loss, advantage) if return_advantage else loss

    @staticmethod
    def diversity(res, gts):
        chencherry = SmoothingFunction()
        return - np.array([sentence_bleu(g, r[0], smoothing_function=chencherry.method1) for g, r in zip(gts, res.values())])


class SelfCriticalWithRepetitionLoss(nn.Module):
    def __init__(self):
        super(SelfCriticalWithRepetitionLoss, self).__init__()

    def forward(self, sample, sample_log_probs, greedy_sample, gts_batch, scorers, vocab, return_advantage=False):
        assert sample.shape[0] == sample_log_probs.shape[0] == greedy_sample.shape[0] == len(gts_batch)

        # Don't look for starting token because is always first given and then discarded, during sampling
        res_sample, res_greedy = get_caps(greedy_sample, sample, vocab, skip_start_token=True)
        reward, advantage = get_self_critical_reward(res_sample, res_greedy, gts_batch, scorers, max_len=sample.size(1))

        # Mask tokens out after <end>, and tokens not generated by the network
        mask = create_mask(sample, vocab)

        accuracy = - (sample_log_probs * reward * mask).sum() / mask.sum()
        diversity = self.repetition(res_sample)

        # gamma = 0.5
        # loss = gamma * accuracy + (1 - gamma) * diversity
        loss = accuracy + diversity
        return (loss, advantage) if return_advantage else loss

    @staticmethod
    def repetition(res):
        unigram, bigram, trigram, cuatrigram = Counter(), Counter(), Counter(), Counter()
        n_tokens = 0
        d1, d2, d3, d4 = 0, 0, 0, 0

        for a, b in res.items():
            cap = b[0].split()
            v_len = len(cap)
            n_tokens += v_len

            unigram.update(cap)
            bigram.update([tuple(cap[i:i + 2]) for i in range(len(cap) - 1)])
            trigram.update([tuple(cap[i:i + 3]) for i in range(len(cap) - 2)])
            cuatrigram.update([tuple(cap[i:i + 4]) for i in range(len(cap) - 3)])

            d1 += ((torch.tensor(list(unigram.values())).float() - 1) ** 2).sum() / v_len if v_len > 0 else 0
            d2 += ((torch.tensor(list(bigram.values())).float() - 1) ** 2).sum() / (v_len - 1) if v_len > 1 else 0
            d3 += ((torch.tensor(list(trigram.values())).float() - 1) ** 2).sum() / (v_len - 2) if v_len > 2 else 0
            d4 += ((torch.tensor(list(cuatrigram.values())).float() - 1) ** 2).sum() / (v_len - 3) if v_len > 3 else 0

            unigram.clear()
            bigram.clear()
            trigram.clear()
            cuatrigram.clear()

        # for a, b in res.items():
        #     cap = b[0].split()
        #     v_len = len(cap)
        #     n_tokens += v_len
        #     unigram.update(cap)
        #     bigram.update([tuple(cap[i:i + 2]) for i in range(len(cap) - 1)])
        #     trigram.update([tuple(cap[i:i + 3]) for i in range(len(cap) - 2)])
        #     cuatrigram.update([tuple(cap[i:i + 4]) for i in range(len(cap) - 3)])
        #
        # d1 = ((np.stack(unigram.values()) - 1) ** 3).sum() / n_tokens
        # d2 = ((np.stack(bigram.values()) - 1) ** 3).sum() / n_tokens
        # d3 = ((np.stack(trigram.values()) - 1) ** 3).sum() / n_tokens
        # d4 = ((np.stack(cuatrigram.values()) - 1) ** 3).sum() / n_tokens

        return d1 * 0.1 + d2 + d3 + d4
        # 1 - len(Counter([tuple(s2[i:i + 2]) for i in range(len(s2) - 1)])) / len(s2)
        # ((np.array(list(Counter([tuple(s1[i:i + 2]) for i in range(len(s1) - 1)]).values())) - 1) ** 3).sum() / len(s1)


class MixedLoss(nn.Module):
    """
    A Deep Reinforced Model for Abstractive Summarization.
    https://arxiv.org/abs/1705.04304.pdf
    Code from https://github.com/ramakanth-pasunuru/video_captioning_rl/blob/master/trainer.py
    """
    def __init__(self):
        super(MixedLoss, self).__init__()
        self.rl = SelfCriticalLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, sample, sample_log_probs, outputs, greedy_sample, gts_batch, scorers, vocab, targets, lengths, gamma_ml_rl, return_advantage=False):
        packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        assert targets.size() != packed_outputs.size(), 'Targets and outputs dont have same dimension. ' \
                                                        'Check sequence length on the unpacked tensors.'

        rl_loss, advantage = self.rl(sample, sample_log_probs, greedy_sample, gts_batch, scorers, vocab, return_advantage=return_advantage)
        ml_loss = self.ce(packed_outputs, targets)

        loss = gamma_ml_rl * rl_loss + (1 - gamma_ml_rl) * ml_loss

        return (loss, advantage) if return_advantage else loss


class FACELoss(nn.Module):
    """
    FACE: Improving Neural Response Diversity with Frequency-Aware Cross-Entropy Loss
    https://arxiv.org/abs/1902.09191.pdf
    Code from https://github.com/ShaojieJiang/FACE
    """
    def __init__(self, vocab_size, frequency_type='out', weighing_time='pre'):
        """
        FACE initialization.
        :param frequency_type: What to use for calculating token frequency.
        :param weighing_time: When to apply weight to losses.
        """
        super(FACELoss, self).__init__()
        assert frequency_type in ['out', 'gt', 'none']
        assert weighing_time in ['pre', 'post', 'none']

        self.frequency_type = frequency_type
        self.weighing_time = weighing_time
        self.criterion = nn.CrossEntropyLoss(reduction='none' if self.weighing_time == 'post' else 'mean')
        self.word_freq = torch.zeros(vocab_size).to(device)

    def forward(self, sample, packed_outputs, targets, packed_targets, vocab):

        # Update token frequency, or not
        if self.frequency_type == 'gt':
            # self.update_frequency(self.clean_preds(targets))
            self.update_frequency(clean_word_ids(targets, vocab))
        elif self.frequency_type == 'out':
            # self.update_frequency(self.clean_preds(sample))
            self.update_frequency(clean_word_ids(sample, vocab))

        # calculate loss w/ or w/o pre-/post-weight
        if self.weighing_time == 'pre':
            self.criterion.weight = self.loss_weight()
            loss = self.criterion(packed_outputs, packed_targets)
        elif self.weighing_time == 'post':
            loss = self.criterion(packed_outputs, packed_targets)
            freq_pred = self.word_freq[sample.view(-1)].float().sum()
            freq_GT = self.word_freq[targets.view(-1)].float().sum()
            total_freq = self.word_freq.sum()
            weight = 1 + F.relu(freq_pred - freq_GT) / total_freq
            loss = torch.matmul(loss, weight)

        # notnull = packed_targets.ne(self.NULL_IDX)
        # target_tokens = notnull.long().sum().item()
        target_tokens = packed_targets.view(-1).size(0)

        return loss / target_tokens

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred.tolist())

        for k, v in curr.items():
            self.word_freq[k] += v

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if self.END_IDX in pred:
                ind = pred.index(self.END_IDX) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            # if pred[0] == self.START_IDX:
            #     pred = pred[1:]
            res.append(pred)
        return res

    def loss_weight(self):
        RF = self.word_freq / self.word_freq.sum()  # relative frequency
        a = -1 / RF.max()
        weight = a * RF + 1
        return weight / weight.sum() * len(weight)  # normalization


class MixedWithFACELoss(nn.Module):
    """
    Mixed: A Deep Reinforced Model for Abstractive Summarization.
    https://arxiv.org/abs/1705.04304.pdf
    Code from https://github.com/ramakanth-pasunuru/video_captioning_rl/blob/master/trainer.py

    FACE: Improving Neural Response Diversity with Frequency-Aware Cross-Entropy Loss
    https://arxiv.org/abs/1902.09191.pdf
    Code from https://github.com/ShaojieJiang/FACE
    """
    def __init__(self, vocab_size):
        super(MixedWithFACELoss, self).__init__()
        self.rl = SelfCriticalLoss()
        self.face = FACELoss(vocab_size)

    def forward(self, sample, sample_log_probs, outputs, greedy_sample, gts_batch, scorers, vocab, captions, targets, lengths, gamma_ml_rl, return_advantage=False):
        packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        assert targets.size() != packed_outputs.size(), 'Targets and outputs dont have same dimension. ' \
                                                        'Check sequence length on the unpacked tensors.'

        rl_loss, advantage = self.rl(sample, sample_log_probs, greedy_sample, gts_batch, scorers, vocab, return_advantage=return_advantage)
        ml_loss = self.face(sample, packed_outputs, targets=captions, packed_targets=targets, vocab=vocab)

        loss = gamma_ml_rl * rl_loss + (1 - gamma_ml_rl) * ml_loss

        return (loss, advantage) if return_advantage else loss

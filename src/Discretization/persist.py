import pandas as pd
import os
import matplotlib.pyplot as plt

"""
Persist Python implementation with choice between Kullback-Leibler divergence and Wasserstein distance and EF or EW initialization.
Author: Lenaig Cornanguer (contact: firstname.lastname[at]irisa.fr)
Copyright (C) 2022  Lenaig Cornanguer
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

import math
import numpy as np

class Persist():
    def __init__(self, x, break_min=2, break_max=10, divergence="w", skip=np.array([4, 4]), candidates="EW"):
        self.x = x
        self.break_min = break_min
        self.break_max = break_max
        self.divergence = divergence # kl: Kullback-Leibler divergence / w: Wasserstein distance
        self.skip = skip
        self.candidates = candidates # EW: equal width / EF: equal frequency
        self.bins = None
        self.pscores = None

        self.persistbins()




    def persistbins(self):
        if self.candidates == "EF":
            candidate_cuts = np.percentile(self.x, range(1, 100), interpolation='midpoint')
        else:
            candidate_cuts = np.arange(np.nanmin(self.x), np.nanmax(self.x), (np.nanmax(self.x) - np.nanmin(self.x)) / 100)
        # keep track of which candidates were used
        free_candidates = np.ones((len(candidate_cuts), 1))
        # exclude first and last few
        free_candidates[
            np.ix_(list(range(self.skip[0])) + list(range(len(candidate_cuts) - self.skip[1], len(candidate_cuts))))] = 0
        # store bins and score for each k
        best_bins = np.full(shape=(self.break_max - self.break_min + 1, self.break_max - 1), fill_value=np.nan)
        best_pscores = np.zeros((self.break_max - self.break_min + 1, 1))
        # start searching cuts
        bins = list()
        for j in range(1, self.break_max):
            if len(np.nonzero(free_candidates)[0]) == 0: break
            # try all free candidate cuts
            current_cuts = np.nonzero(free_candidates)[0]
            if len(current_cuts) > 0:
                pscores = np.zeros((len(current_cuts), 1))
                for i in range(len(current_cuts)):
                    pscores[i] = self.persistence(np.sort(
                        np.concatenate((bins, candidate_cuts[current_cuts[i]]), axis=None)), divergence=self.divergence)
            # pick the one with best persistence
            best_score, best_ind = pscores.max(axis=0), pscores.argmax(axis=0)
            best_cut = candidate_cuts[current_cuts[best_ind]]
            bins = np.concatenate((bins, best_cut))
            free_candidates[range(current_cuts[best_ind][0] - 4, current_cuts[best_ind][0] + 4 + 1)] = 0
            # store result if in requested range of k
            if j + 1 >= self.break_min:
                best_pscores[j - (self.break_min - 1)] = best_score
                if len(bins) < self.break_max:
                    best_bins[j - (self.break_min - 1)] = np.concatenate(
                        (bins, np.full(shape=(self.break_max - len(bins) - 1), fill_value=np.nan)))
                else:
                    best_bins[j - (self.break_min - 1)] = bins
        # pick best if several k were tried
        if self.break_min != self.break_max:
            dummy, best = best_pscores.max(axis=0), best_pscores.argmax(axis=0)
            # bins = best_bins[best]
            bins = best_bins

        self.bins = bins
        self.pscores = best_pscores

    def persistence(self, bins, divergence):
        y = self.discretize(bins)
        p, a, k = self.mcml(y)
        st = np.diag(a)  # self-transitions
        if divergence == "w":
            pscore = np.mean((-1) ** (st < np.transpose(p)) * self.wasserstein(
                np.concatenate((st.reshape(-1, 1), 1 - st.reshape(-1, 1)), axis=1),
                np.concatenate([p.reshape(-1, 1), 1 - p.reshape(-1, 1)], axis=1)))
        else: # kl
            pscore = np.mean((-1) ** (st < np.transpose(p)) * self.kldiv(
                np.concatenate((st.reshape(-1, 1), 1 - st.reshape(-1, 1)), axis=1),
                np.concatenate([p.reshape(-1, 1), 1 - p.reshape(-1, 1)], axis=1), 1))
        return pscore

    def mcml(self, x, states=None):
        x = np.transpose(x)

        if not states: states = np.array(range(int(np.max(x)) + 1))

        # number of states and symbols
        k = max(states) + 1
        n = x.shape[0]

        if x.shape[1] == 1:
            # estimate start probabilities
            p = sum(np.tile(x, (1, k)) == np.tile(np.transpose(states), (n, 1))) / n
            # estimate transition probabilities
            obstup = np.concatenate((np.array(x[0:n - 1]), x[1: n]), axis=1)
        else:
            # for multiple ts
            # estimate start probabilities
            symb = np.swapaxes(np.array([x]*k), 1, 2)
            for i in range(k):
                symb[i] = i
            p = np.sum(np.sum(np.swapaxes(np.array([x]*k), 1, 2) == symb, axis=1), axis = 1)/ x.size
            u = np.swapaxes(np.stack((np.array(x[0:n - 1]), x[1: n]), axis=1), 1, 2)
            obstup = np.array([uzz for uz in u for uzz in uz])

        t, c = np.unique(obstup, return_counts=True, axis=0)
        a = np.zeros(shape=(k, k))
        for i in range(t.shape[0]):
            a[int(t[i, 0]), int(t[i, 1])] = c[i]

        with np.errstate(invalid='ignore'):
            a = a / np.tile(np.reshape(np.sum(a, 1), (k, 1)), (1, a.shape[1]))
        a = np.where(np.isnan(a), 0, a)

        return (p, a, k)

    def kldiv(self, p, q, sym=True):
        # p: self-transitions vs non-self probabilities
        # q: marginal probabilities
        # not defined for zero probabilities, replace with very small values
        eps = 1 / 1000000
        for i in range(p.shape[0]):
            z = p[i, :] == 0
            p[i, np.nonzero(z)] = eps
            p[i, np.nonzero(np.logical_not(z))] = p[i, np.nonzero(np.logical_not(z))] - (
                    sum(z) * eps / (p.shape[1] - sum(z)))
        for i in range(q.shape[0]):
            z = q[i, :] == 0
            q[i, np.nonzero(z)] = eps
            q[i, np.nonzero(np.logical_not(z))] = q[i, np.nonzero(np.logical_not(z))] - (
                    sum(z) * eps / (q.shape[1] - sum(z)))

        d = np.mean(p * np.log(p / q), axis=1)
        if sym:
            d = 0.5 * (d + np.mean(q * np.log(q / p), axis=1))
        return d

    def wasserstein(self, p, q):
        res = np.zeros(shape=len(p))
        for i in range(len(p)):
            res[i] = abs(p[i][0] - q[i][0])
        return res

    def discretize(self, bins):
        if self.x.shape[1] == 1:
            y = np.zeros((self.x.shape[0], 1))
        else:
            y = np.zeros(self.x.shape, dtype=int)
        bins = np.concatenate((-math.inf, np.sort(bins), math.inf), axis=None)
        for i in range(len(bins) - 1):
            y[np.nonzero(np.logical_and(self.x >= bins[i], self.x < bins[i + 1]))] = i
        return y

def discretize_traces_with_bins(traces, bins):
    discretized_traces = []

    for trace in traces:
        values = [v for v, t in trace]
        times = [t for v, t in trace]

        labels = np.digitize(values, bins, right=False)

        # shift to 0-based
        labels = labels - 1

        # safety clamp
        labels = np.clip(labels, 0, len(bins) - 2)

        discretized_trace = [
            (int(label), int(time))
            for label, time in zip(labels, times)
        ]

        discretized_traces.append(discretized_trace)

    return discretized_traces

def get_best_bins(persist_obj, ts):
    best_idx = np.argmax(persist_obj.pscores)

    bins = persist_obj.bins[best_idx]
    bins = bins[~np.isnan(bins)]
    bins = np.sort(bins)

    # use the real data range as outer bin edges
    bins = np.concatenate(([np.min(ts)], bins, [np.max(ts)]))
    return bins

# def get_best_bins(persist_obj, ts):
#
#     print("\n--- PERSISTENCE DEBUG ---")
#     print("pscores:", persist_obj.pscores)
#     print("num candidates:", len(persist_obj.pscores))
#
#     for i, (bins, score) in enumerate(zip(persist_obj.bins, persist_obj.pscores)):
#         if bins is None:
#             continue
#         clean_bins = bins[~np.isnan(bins)]
#         print(f"i={i} | bins={len(clean_bins)} | score={score}")
#
#     best_idx = np.argmax(persist_obj.pscores)
#     print("\nBEST idx:", best_idx)
#     print("BEST score:", persist_obj.pscores[best_idx])
#     print("------------------------\n")
#
#     bins = persist_obj.bins[best_idx]
#     bins = bins[~np.isnan(bins)]
#     bins = np.sort(bins)
#
#     bins = np.concatenate(([np.min(ts)], bins, [np.max(ts)]))
#
#     return bins

def flatten_traces_to_ts(data_lists):
    ts = np.concatenate([
        np.array([v for v, t in trace])
        for trace in data_lists
    ])
    return np.asarray(ts).reshape(-1, 1)


def plot_and_save_breakpoints(ts, bins, save_path, show=True):
    plt.figure()

    plt.plot(ts.flatten())
    for b in bins:
        plt.axhline(b, color="red")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
#
# def sax_discretization_multi(data_lists, w, k):
#     breakpoints = norm.ppf(np.linspace(0, 1, k + 1)[1:-1])
#
#     def znorm(v):
#         sigma = v.std()
#         return (v - v.mean()) / sigma if sigma != 0 else np.zeros_like(v)
#
#     def paa(v, t, w):
#         v_segs = np.array_split(v, w)
#         t_segs = np.array_split(t, w)
#         return (
#             np.array([seg.mean() for seg in v_segs]),
#             np.array([int(seg.mean()) for seg in t_segs])
#         )
#
#     discretized = []
#     for trace in data_lists:
#         v = np.array([val for val, _ in trace])
#         t = np.array([time for _, time in trace])
#         paa_v, paa_t = paa(znorm(v), t, w)
#         labels = np.digitize(paa_v, breakpoints, right=False)
#         discretized.append([(int(l), int(ts)) for l, ts in zip(labels, paa_t)])
#
#
#
#
#     return discretized, breakpoints

#
# def sax_discretization_multi(data_lists, w, k):
#     breakpoints = norm.ppf(np.linspace(0, 1, k + 1)[1:-1])
#
#     def znorm(v):
#         sigma = v.std()
#         return (v - v.mean()) / sigma if sigma != 0 else np.zeros_like(v)
#
#     def paa(v, t, w):
#         v_segs = np.array_split(v, w)
#         t_segs = np.array_split(t, w)
#         return (
#             np.array([seg.mean() for seg in v_segs]),
#             np.array([int(seg.mean()) for seg in t_segs])
#         )
#
#     discretized = []
#     all_norm_vals = []
#
#     for trace in data_lists:
#         v = np.array([val for val, _ in trace])
#         t = np.array([time for _, time in trace])
#
#         norm_v = znorm(v)
#         all_norm_vals.extend(norm_v)
#
#         paa_v, paa_t = paa(norm_v, t, w)
#         labels = np.digitize(paa_v, breakpoints, right=False)
#
#         discretized.append([(int(l), int(ts)) for l, ts in zip(labels, paa_t)])
#
#     bins = np.concatenate((
#         [np.min(all_norm_vals)],
#         breakpoints,
#         [np.max(all_norm_vals)]
#     ))
#
#     return discretized, bins

#
# def sax_discretization_multi(data_lists, w, k):
#     breakpoints = norm.ppf(np.linspace(0, 1, k + 1)[1:-1])
#
#     # Compute global stats once across all traces
#     all_v = np.concatenate([np.array([val for val, _ in trace]) for trace in data_lists])
#     global_mean = all_v.mean()
#     global_std = all_v.std() if all_v.std() != 0 else 1.0
#
#     def paa(v, t, w):
#         v_segs = np.array_split(v, w)
#         t_segs = np.array_split(t, w)
#         return (
#             np.array([seg.mean() for seg in v_segs]),
#             np.array([int(seg.mean()) for seg in t_segs])
#         )
#
#     discretized = []
#     all_norm_vals = []
#
#     for trace in data_lists:
#         v = np.array([val for val, _ in trace])
#         t = np.array([time for _, time in trace])
#         norm_v = (v - global_mean) / global_std  # global normalization
#         all_norm_vals.extend(norm_v)
#         paa_v, paa_t = paa(norm_v, t, w)
#         labels = np.digitize(paa_v, breakpoints, right=False)
#         discretized.append([(int(l), int(ts)) for l, ts in zip(labels, paa_t)])
#
#     bins = np.concatenate((
#         [np.min(all_norm_vals)],
#         breakpoints,
#         [np.max(all_norm_vals)]
#     ))
#
#     return discretized, bins, global_mean, global_std  # return stats for inverse transform

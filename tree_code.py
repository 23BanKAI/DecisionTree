import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    assert len(feature_vector) == len(target_vector)

    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    diff_indices = np.where(feature_sorted[:-1] != feature_sorted[1:])[0]
    if len(diff_indices) == 0:
        return np.array([]), np.array([]), None, np.inf

    thresholds = (feature_sorted[diff_indices] + feature_sorted[diff_indices + 1]) / 2
    ginis = []

    n_total = len(target_sorted)

    for threshold in thresholds:
        left_mask = feature_sorted < threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            ginis.append(np.inf)
            continue

        p1_left = np.mean(target_sorted[left_mask])
        p0_left = 1 - p1_left
        h_left = 1 - p1_left**2 - p0_left**2

        p1_right = np.mean(target_sorted[right_mask])
        p0_right = 1 - p1_right
        h_right = 1 - p1_right**2 - p0_right**2

        gini = (left_mask.sum() / n_total) * h_left + (right_mask.sum() / n_total) * h_right
        ginis.append(gini)

    ginis = np.array(ginis)
    best_idx = np.argmin(ginis)

    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("Неизвестный тип признака")

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._tree = {}

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = int(sub_y[0])
            return

        if (self._max_depth is not None and depth >= self._max_depth) or len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            else:  # categorical
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {k: clicks.get(k, 0) / counts[k] for k in counts}
                sorted_cats = sorted(ratio, key=ratio.get)
                categories_map = {cat: i for i, cat in enumerate(sorted_cats)}
                feature_vector = np.array([categories_map.get(v, -1) for v in sub_X[:, feature]])

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None or gini is None:
                continue

            if gini_best is None or gini < gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold

                if feature_type == "real":
                    split = feature_vector < threshold
                    categories_map_best = None
                else:
                    split = feature_vector < threshold
                    categories_map_best = categories_map

        if feature_best is None or split.sum() < self._min_samples_leaf or (~split).sum() < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        feature_type = self._feature_types[feature_best]

        if feature_type == "real":
            node["threshold"] = float(threshold_best)
        else:
            node["categories_split"] = [k for k, v in categories_map_best.items() if v < threshold_best]

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            threshold = node.get("threshold")
            return self._predict_node(
                x, node["left_child"] if float(x[feature_idx]) < threshold else node["right_child"]
            )
        else:
            category = x[feature_idx]
            left_categories = node.get("categories_split", [])
            return self._predict_node(
                x, node["left_child"] if category in left_categories else node["right_child"]
            )

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X], dtype=int)

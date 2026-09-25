"""M1: Weight-of-Evidence logistic-regression scorecard.

The industry-standard, fully explainable credit model. Every applicant's score is a
sum of points, one per characteristic, that can be printed on a single page.

1. Bin each feature on TRAIN: up to 10 quantile bins, each >= 5% of loans, missing
   values in their own bin. Where business sense says risk should move one way
   (higher FICO = safer, higher DTI = riskier, ...), adjacent bins are merged
   until the bad rate is monotone.
2. Replace each bin by its Weight of Evidence:  WoE = ln(% of goods in bin / % of bads in bin).
   Positive WoE = safer than average.
3. Keep features with Information Value (IV) >= 0.02; of any pair of WoE columns
   correlated above 0.7, drop the one with lower IV.
4. L2 logistic regression on the WoE columns, regularization C tuned on VALIDATION AUC.
5. Scale to points: 600 points at 50:1 good:bad odds; every 20 points doubles the odds.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MISSING = "Missing"


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def _bin_index(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin i covers (edges[i-1], edges[i]]; first bin is (-inf, edges[0]], last is (edges[-1], inf)."""
    return np.searchsorted(edges, x, side="left")


def _merge_small_bins(x, edges, min_count):
    """Remove cut points until every bin holds at least min_count loans."""
    while edges:
        counts = np.bincount(_bin_index(x, edges), minlength=len(edges) + 1)
        smallest = int(np.argmin(counts))
        if counts[smallest] >= min_count:
            break
        if smallest == 0:
            edges.pop(0)
        elif smallest == len(counts) - 1:
            edges.pop(-1)
        else:  # merge with the smaller neighbour
            edges.pop(smallest - 1 if counts[smallest - 1] <= counts[smallest + 1] else smallest)
    return edges


def _merge_until_monotone(x, y, edges):
    """Merge adjacent bins until the bad rate only rises (or only falls) across bins."""
    direction = np.sign(spearmanr(x, y).statistic) or 1
    while edges:
        idx = _bin_index(x, edges)
        bad_rate = np.bincount(idx, weights=y, minlength=len(edges) + 1) / np.bincount(idx, minlength=len(edges) + 1)
        violations = np.where(np.diff(bad_rate) * direction < 0)[0]
        if len(violations) == 0:
            break
        edges.pop(int(violations[0]))   # merges bin i with bin i+1
    return edges


@dataclass
class FeatureBins:
    feature: str
    edges: list | None = None      # numeric features: cut points
    groups: dict | None = None     # categorical features: level -> bin label
    table: pd.DataFrame | None = None
    iv: float = 0.0

    def labels(self, x: pd.Series) -> pd.Series:
        if self.groups is not None:
            return x.astype(object).map(self.groups).fillna(MISSING)
        values = x.to_numpy(dtype=float)
        names = self._numeric_names()
        labels = np.array(names, dtype=object)[_bin_index(np.nan_to_num(values, nan=0.0), self.edges)]
        labels[np.isnan(values)] = MISSING
        return pd.Series(labels, index=x.index)

    def _numeric_names(self) -> list[str]:
        e = self.edges
        if not e:
            return ["All values"]
        return ([f"<= {_fmt(e[0])}"] + [f"({_fmt(lo)}, {_fmt(hi)}]" for lo, hi in zip(e[:-1], e[1:])]
                + [f"> {_fmt(e[-1])}"])

    def woe(self, x: pd.Series) -> np.ndarray:
        """Bins never seen in train (e.g. a new category) get WoE 0 = average risk."""
        lookup = self.table.set_index("bin")["woe"]
        return self.labels(x).map(lookup).fillna(0.0).to_numpy()


def _woe_table(labels: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, float]:
    t = pd.DataFrame({"bin": labels.values, "bad": y.values}).groupby("bin", sort=False)["bad"].agg(["size", "sum"])
    t.columns = ["n", "n_bad"]
    t["n_good"] = t["n"] - t["n_bad"]
    # +0.5 smoothing keeps WoE finite for bins with no goods or no bads
    dist_good = (t["n_good"] + 0.5) / (t["n_good"] + 0.5).sum()
    dist_bad = (t["n_bad"] + 0.5) / (t["n_bad"] + 0.5).sum()
    t["share"] = t["n"] / t["n"].sum()
    t["bad_rate"] = t["n_bad"] / t["n"]
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv_part"] = (dist_good - dist_bad) * t["woe"]
    return t.reset_index(), float(t["iv_part"].sum())


def fit_feature_bins(x: pd.Series, y: pd.Series, sc: dict, monotone: bool) -> FeatureBins:
    bins = FeatureBins(feature=x.name)
    if isinstance(x.dtype, pd.CategoricalDtype):
        share = x.value_counts(normalize=True)
        bins.groups = {lvl: (str(lvl) if share[lvl] >= sc["min_category_share"] else "Rare levels")
                       for lvl in x.cat.categories}
    else:
        present = x.notna().to_numpy()
        xs, ys = x.to_numpy(dtype=float)[present], y.to_numpy(dtype=float)[present]
        quantiles = np.linspace(0, 1, sc["max_bins"] + 1)[1:-1]
        edges = sorted(set(np.quantile(xs, quantiles)))
        edges = [e for e in edges if e < xs.max()]                    # a cut at the max makes an empty bin
        edges = _merge_small_bins(xs, edges, sc["min_bin_share"] * len(x))
        if monotone:
            edges = _merge_until_monotone(xs, ys, edges)
        bins.edges = edges

    bins.table, bins.iv = _woe_table(bins.labels(x), y)
    order = {name: i for i, name in enumerate((bins._numeric_names() if bins.groups is None
                                                else list(dict.fromkeys(bins.groups.values()))) + [MISSING])}
    bins.table = bins.table.sort_values("bin", key=lambda s: s.map(order)).reset_index(drop=True)
    return bins


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

class WoEScorecard:

    def fit(self, X_train, y_train, X_valid, y_valid, cfg: dict) -> "WoEScorecard":
        sc = cfg["scorecard"]
        self.cfg = sc
        y_train = pd.Series(np.asarray(y_train), index=X_train.index)
        self.bins = {col: fit_feature_bins(X_train[col], y_train, sc, monotone=col in sc["monotone_features"])
                     for col in X_train.columns}
        self.features, self.selection = self._select_features(X_train)
        self._drop_wrong_signs(X_train, y_train)

        # tune C on validation AUC (train is only used to fit coefficients)
        W_train, W_valid = self.woe_matrix(X_train), self.woe_matrix(X_valid)
        self.c_search = []
        for c in sc["c_grid"]:
            model = LogisticRegression(C=c, max_iter=2000).fit(W_train, y_train)
            self.c_search.append({"C": c, "valid_auc": roc_auc_score(y_valid, model.predict_proba(W_valid)[:, 1])})
        self.best_c = max(self.c_search, key=lambda r: r["valid_auc"])["C"]
        self.model = LogisticRegression(C=self.best_c, max_iter=2000).fit(W_train, y_train)
        self.points_table = self._points_table()
        return self

    def _select_features(self, X_train):
        """IV filter, then drop the weaker feature of any highly correlated pair."""
        sc = self.cfg
        iv = pd.Series({f: b.iv for f, b in self.bins.items()}).sort_values(ascending=False)
        W = pd.DataFrame({f: self.bins[f].woe(X_train[f]) for f in iv.index})
        corr = W.corr().abs()
        kept, rows = [], []
        for f in iv.index:
            if iv[f] < sc["min_iv"]:
                reason = f"IV < {sc['min_iv']}"
            elif kept and corr.loc[f, kept].max() > sc["max_woe_corr"]:
                partner = corr.loc[f, kept].idxmax()
                reason = f"|corr| {corr.loc[f, partner]:.2f} with {partner} (higher IV)"
            else:
                reason = ""
                kept.append(f)
            rows.append({"feature": f, "iv": iv[f], "selected": reason == "", "drop_reason": reason})
        return kept, pd.DataFrame(rows)

    def _drop_wrong_signs(self, X_train, y_train) -> None:
        """WoE is 'higher = safer', so every coefficient must be negative.

        A positive one means the feature's effect flipped once the others are in the
        model (collinearity) - its points would reward risk. Drop the lowest-IV
        offender and refit until every sign is right.
        """
        while True:
            model = LogisticRegression(C=1.0, max_iter=2000).fit(self.woe_matrix(X_train), y_train)
            wrong = [f for f, c in zip(self.features, model.coef_[0]) if c > 0]
            if not wrong:
                return
            weakest = min(wrong, key=lambda f: self.bins[f].iv)
            self.features.remove(weakest)
            row = self.selection["feature"] == weakest
            self.selection.loc[row, ["selected", "drop_reason"]] = [False, "wrong sign in multivariate fit"]

    def woe_matrix(self, X) -> pd.DataFrame:
        return pd.DataFrame({f: self.bins[f].woe(X[f]) for f in self.features}, index=X.index)

    def predict_proba(self, X) -> np.ndarray:
        """Probability of default (uncalibrated)."""
        return self.model.predict_proba(self.woe_matrix(X))[:, 1]

    def _points_table(self) -> pd.DataFrame:
        """Points per bin so that total points = 600 at 50:1 odds, +20 points = 2x the odds."""
        sc = self.cfg
        factor = sc["pdo"] / np.log(2)
        offset = sc["base_score"] - factor * np.log(sc["base_odds"])
        k, intercept = len(self.features), self.model.intercept_[0]
        rows = []
        for f, coef in zip(self.features, self.model.coef_[0]):
            t = self.bins[f].table.copy()
            t.insert(0, "feature", f)
            t["iv"] = self.bins[f].iv
            t["coefficient"] = coef
            # logit(PD) = intercept + sum(coef * WoE);  score = offset - factor * logit(PD)
            t["points"] = offset / k - factor * (coef * t["woe"] + intercept / k)
            rows.append(t[["feature", "bin", "n", "share", "bad_rate", "woe", "iv", "coefficient", "points"]])
        return pd.concat(rows, ignore_index=True)

    def score_points(self, X) -> np.ndarray:
        total = np.zeros(len(X))
        for f in self.features:
            lookup = self.points_table.query("feature == @f").set_index("bin")["points"]
            neutral = lookup.mean()   # unseen bin -> average points for that feature
            total += self.bins[f].labels(X[f]).map(lookup).fillna(neutral).to_numpy()
        return total

    def summary(self) -> dict:
        coefs = dict(zip(self.features, self.model.coef_[0]))
        return {
            "n_candidate_features": len(self.bins),
            "n_selected_features": len(self.features),
            "selected_features": self.features,
            "best_C": self.best_c,
            "c_search": self.c_search,
            # WoE is 'higher = safer', so every coefficient should be negative
            "wrong_sign_coefficients": [f for f, c in coefs.items() if c > 0],
        }

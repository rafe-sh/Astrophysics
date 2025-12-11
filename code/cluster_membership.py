"""Cluster membership determination pipeline for Gaia DR3 open clusters.

This module implements a reproducible, multi-stage membership analysis
following contemporary Gaia DR3 quality prescriptions:

* Parallax zero-point correction following Lindegren et al. (2021).
* Photometric flux-excess cleaning using the C* diagnostic of Riello et al. (2021).
* Coarse geometric filtering, spatial overdensity selection via an MST,
  sigma clipping, and probabilistic classification with a Gaussian Mixture Model.

The goal is to provide a publication-grade, modular pipeline suitable for
Galactic Archaeology and star cluster dynamical studies.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:  # scienceplots is optional; fallback to default styles if unavailable
    import scienceplots

    _HAS_SCIENCEPLOTS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SCIENCEPLOTS = False
from scipy.sparse import csgraph
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

try:  # Optional: real Lindegren zero-point tables if available
    from zero_point import zpt

    zpt.load_tables()
    _HAS_ZPT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_ZPT = False


_REQUIRED_COLUMNS = {
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "parallax_error",
    "pmra_error",
    "pmdec_error",
    "phot_g_mean_mag",
    "bp_rp",
    "phot_bp_rp_excess_factor",
    "ruwe",
    "visibility_periods_used",
    "matched_transits",
}


@dataclass
class ClusterMembership:
    """Hybrid open cluster membership pipeline for Gaia DR3.

    The pipeline closely follows the recommended order of operations to isolate
    high-confidence members. Each stage can be called independently, but the
    typical workflow is::

        cm = ClusterMembership(df, center, tolerances)
        cm.preprocess()
        cm.coarse_filter()
        cm.apply_mst()
        cm.apply_sigma_clip()
        cm.apply_gmm()
        members = cm.get_members()

    Parameters
    ----------
    data : pandas.DataFrame
        Gaia DR3 catalogue subset. Must contain the standard astrometric and
        photometric columns (``ra``, ``dec``, ``parallax``, ``pmra``, ``pmdec``,
        errors, ``phot_g_mean_mag``, ``bp_rp``, ``phot_bp_rp_excess_factor``).
    center : tuple[float, float, float, float, float]
        Cluster centroid in (RA [deg], Dec [deg], parallax [mas], pmra [mas/yr],
        pmdec [mas/yr]). Used for coarse geometric filtering.
    tolerances : tuple[float, float, float, float, float]
        Half-widths for the coarse filter in the same order as ``center``. Units
        are degrees for RA/Dec, milliarcseconds for parallax, and mas/yr for
        proper motions.
    mst_edge_threshold : float, optional
        Threshold on standardized MST edge length. Edges longer than this are
        removed to isolate the main spatial overdensity. Defaults to 0.31.
    random_state : int, optional
        Random seed for the Gaussian Mixture Model. Defaults to 42.
    """

    data: pd.DataFrame
    center: tuple[float, float, float, float, float]
    tolerances: tuple[float, float, float, float, float]
    mst_edge_threshold: float = 0.31
    random_state: int = 42

    def __post_init__(self) -> None:
        self.data = self.data.copy()
        missing = _REQUIRED_COLUMNS.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required Gaia columns: {sorted(missing)}")

        self.scaler_spatial = StandardScaler()
        self.scaler_full = StandardScaler()
        self._mst_mask: Optional[np.ndarray] = None
        self._sigma_mask: Optional[np.ndarray] = None
        self._gmm: Optional[GaussianMixture] = None
        self._gmm_labels: Optional[np.ndarray] = None
        self._probabilities: Optional[np.ndarray] = None
        self._preprocessed: Optional[pd.DataFrame] = None
        self._coarse: Optional[pd.DataFrame] = None
        self._mst_selected: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Helper corrections
    @staticmethod
    def lindegren_zpt(parallax: pd.Series, g_mag: pd.Series, bp_rp: pd.Series) -> pd.Series:
        """Apply the Gaia DR3 parallax zero-point correction.

        If the optional :mod:`zero_point` package is installed, this function
        uses the official Lindegren et al. (2021) look-up tables. Otherwise, a
        compact colour-magnitude polynomial is used to approximate the bias for
        the five-parameter solution (Table 3 of Lindegren+2021).
        """

        if _HAS_ZPT:
            return parallax - zpt.get_zpt(
                phot_g_mean_mag=g_mag.values,
                color=bp_rp.values,
                nu_eff_used_in_astrometry=np.full_like(bp_rp.values, np.nan),
                pseudocolour=np.full_like(bp_rp.values, np.nan),
                ecl_lat=np.full_like(bp_rp.values, np.nan),
                astrometric_params_solved=np.full_like(bp_rp.values, 31),
            )

        c = bp_rp.clip(-0.2, 3.5)
        g = g_mag.clip(6, 19)
        zpt_poly = -0.017 - 0.0032 * (c - 1.5) - 0.0006 * (g - 12)
        return parallax - zpt_poly

    @staticmethod
    def riello_cstar(excess_factor: pd.Series, bp_rp: pd.Series) -> pd.Series:
        """Riello et al. (2021) flux-excess corrected C* diagnostic.

        The corrected C* quantity removes the intrinsic colour trend of the
        flux excess factor. Stars with |C*| > 5 sigma(C*) are photometrically
        unreliable (often affected by crowding or contamination) and should be
        excluded.
        """

        bp_rp = bp_rp.astype(float)
        excess = excess_factor.astype(float)

        c_star = np.zeros_like(bp_rp)
        mask_blue = bp_rp < 0.5
        mask_green = (bp_rp >= 0.5) & (bp_rp < 4.0)
        mask_red = bp_rp >= 4.0

        c_star[mask_blue] = excess[mask_blue] - (
            1.154360 + 0.033772 * bp_rp[mask_blue] + 0.032277 * bp_rp[mask_blue] ** 2
        )
        c_star[mask_green] = excess[mask_green] - (
            1.162004
            + 0.011464 * bp_rp[mask_green]
            + 0.049255 * bp_rp[mask_green] ** 2
            - 0.005879 * bp_rp[mask_green] ** 3
        )
        c_star[mask_red] = excess[mask_red] - (1.057572 + 0.140537 * bp_rp[mask_red])
        return pd.Series(c_star, index=bp_rp.index)

    @staticmethod
    def riello_cstar_sigma(g_mag: pd.Series) -> pd.Series:
        """Intrinsic scatter for the Riello C* diagnostic (Eq. 18)."""

        g = g_mag.astype(float)
        return 0.0059898 + 8.817481e-12 * (g ** 7.618399)

    # ------------------------------------------------------------------
    def preprocess(self) -> pd.DataFrame:
        """Apply Gaia DR3 quality cuts and corrections.

        Steps
        -----
        1. Parallax zero-point correction (Lindegren et al. 2021).
        2. Flux-excess cleaning using C* (Riello et al. 2021), removing stars
           with |C*| > 5 sigma(C*).
        3. Require parallax_over_error >= 5 to retain reliable astrometry.

        Returns
        -------
        pandas.DataFrame
            Quality-controlled catalogue with corrected parallax and C* values.
        """

        df = self.data.copy()
        if df.empty:
            warnings.warn("Input catalogue is empty; skipping preprocessing.")
            self.data = df
            return df

        df = df[
            (df["phot_g_mean_mag"] < 21)
            & (df["ruwe"] < 1.4)
            & (df["visibility_periods_used"] > 9)
            & (df["matched_transits"] > 5)
        ]

        df["parallax_corr"] = self.lindegren_zpt(df["parallax"], df["phot_g_mean_mag"], df["bp_rp"])
        df["c_star"] = self.riello_cstar(df["phot_bp_rp_excess_factor"], df["bp_rp"])
        sigma_c = self.riello_cstar_sigma(df["phot_g_mean_mag"])
        df = df[np.abs(df["c_star"]) <= 5 * sigma_c]

        df["parallax_over_error"] = df["parallax_corr"] / df["parallax_error"].replace(0, np.nan)
        df = df[df["parallax_over_error"] >= 5]
        df = df.dropna(subset=["parallax_corr", "parallax_error", "pmra", "pmdec"])

        if df.empty:
            warnings.warn("Catalogue empty after preprocessing cuts.")

        self.data = df
        self._preprocessed = df.copy()
        return self.data

    def coarse_filter(self) -> pd.DataFrame:
        """Coarse geometric filtering around a supplied cluster center.

        This stage removes obvious background contamination using a hyper-box in
        (RA, Dec, parallax, pmra, pmdec).
        """

        if self.data.empty:
            warnings.warn("No data available for coarse filtering.")
            return self.data

        ra0, dec0, plx0, pmra0, pmdec0 = self.center
        dra, ddec, dplx, dpmra, dpmdec = self.tolerances

        df = self.data[
            (np.abs(self.data["ra"] - ra0) <= dra)
            & (np.abs(self.data["dec"] - dec0) <= ddec)
            & (np.abs(self.data["parallax_corr"] - plx0) <= dplx)
            & (np.abs(self.data["pmra"] - pmra0) <= dpmra)
            & (np.abs(self.data["pmdec"] - pmdec0) <= dpmdec)
        ]

        if df.empty:
            warnings.warn("Catalogue empty after coarse filter.")

        self.data = df
        self._coarse = df.copy()
        return self.data

    def apply_mst(self) -> pd.DataFrame:
        """Spatial Minimum Spanning Tree selection in (RA, Dec, parallax).

        The MST is computed on standardized coordinates to homogenize the units.
        Edges longer than ``mst_edge_threshold`` are cut and only the largest
        connected component is retained as the spatial overdensity associated
        with the cluster.
        """

        if self.data.empty:
            warnings.warn("No data available for MST.")
            return self.data

        coords = self.data[["ra", "dec", "parallax_corr"]].to_numpy()
        scaled = self.scaler_spatial.fit_transform(coords)

        dist_matrix = pairwise_distances(scaled, metric="euclidean")
        mst_sparse = csgraph.minimum_spanning_tree(dist_matrix)
        mst = mst_sparse.toarray().astype(float)

        # Symmetrize and remove long edges
        mst = np.maximum(mst, mst.T)
        mst[mst > self.mst_edge_threshold] = 0.0

        n_components, labels = csgraph.connected_components(mst)
        if n_components == 0:
            warnings.warn("MST produced zero components; retaining all stars.")
            self._mst_mask = np.ones(len(self.data), dtype=bool)
            return self.data

        # Keep the largest component (spatial overdensity)
        largest_label = np.bincount(labels).argmax()
        mask = labels == largest_label
        self._mst_mask = mask
        self.data = self.data.loc[mask].copy()
        self._mst_selected = self.data.copy()
        return self.data

    def apply_sigma_clip(self, sigma: float = 3.0) -> pd.DataFrame:
        """Multivariate sigma clipping in 5D astrometric space.

        Parameters
        ----------
        sigma : float, optional
            Clipping threshold in standard deviations. Default is 3-sigma.
        """

        if self.data.empty:
            warnings.warn("No data available for sigma clipping.")
            return self.data

        features = self.data[["ra", "dec", "parallax_corr", "pmra", "pmdec"]]
        scaled = self.scaler_full.fit_transform(features)
        z_scores = np.abs(scaled)
        mask = (z_scores <= sigma).all(axis=1)

        self._sigma_mask = mask.to_numpy() if isinstance(mask, pd.Series) else mask
        self.data = self.data.loc[mask].copy()

        if self.data.empty:
            warnings.warn("Catalogue empty after sigma clipping.")

        return self.data

    def apply_gmm(self, prob_threshold: float = 0.8) -> pd.DataFrame:
        """Gaussian Mixture Model classification in 5D astrometric space.

        A two-component GMM separates the cluster from the field. The cluster
        component is identified as the one with the smallest positional
        covariance (i.e., the most compact distribution).
        """

        if self.data.empty:
            warnings.warn("No data available for GMM.")
            return self.data

        features = self.data[["ra", "dec", "parallax_corr", "pmra", "pmdec"]]
        scaled = self.scaler_full.fit_transform(features)

        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=self.random_state)
        gmm.fit(scaled)
        responsibilities = gmm.predict_proba(scaled)

        # Identify the compact component as the cluster
        cov_traces = np.array([np.trace(cov) for cov in gmm.covariances_])
        cluster_label = cov_traces.argmin()

        self._gmm = gmm
        self._gmm_labels = np.argmax(responsibilities, axis=1)
        self._probabilities = responsibilities[:, cluster_label]

        self.data["membership_probability"] = self._probabilities
        self.data["gmm_label"] = self._gmm_labels
        self.data["is_member"] = self.data["membership_probability"] >= prob_threshold
        return self.data

    def get_members(self, pmin: float = 0.8) -> pd.DataFrame:
        """Return high-confidence cluster members.

        Parameters
        ----------
        pmin : float, optional
            Minimum cluster membership probability. Default is 0.8.
        """

        if self.data.empty or self._probabilities is None:
            warnings.warn("GMM probabilities not available; did you run apply_gmm()?")
            return pd.DataFrame(columns=self.data.columns)

        members = self.data[self.data["membership_probability"] >= pmin].copy()
        self.data.loc[members.index, "is_member"] = True
        return members

    def get_plotting_data(self, pmin: float = 0.8) -> pd.DataFrame:
        """Assemble a labelled catalogue for visualization.

        Populations
        -----------
        Field
            Stars that pass the coarse filter but were not retained by the MST.
        MST candidate
            Stars inside the MST-connected overdensity before GMM classification.
        High-prob member
            Stars with membership probability >= ``pmin`` from the GMM.
        """

        if self._coarse is None:
            warnings.warn("Coarse-filtered catalogue unavailable; run coarse_filter().")
            return pd.DataFrame()

        plot_df = self._coarse.copy()
        plot_df["population"] = "Field"

        if self._mst_selected is not None:
            idx = plot_df.index.intersection(self._mst_selected.index)
            plot_df.loc[idx, "population"] = "MST candidate"

        if self._probabilities is not None:
            gmm_mask = self.data["membership_probability"] >= pmin
            member_idx = plot_df.index.intersection(self.data.loc[gmm_mask].index)
            plot_df.loc[member_idx, "population"] = "High-prob member"

        return plot_df


class ScientificClusterPipeline:
    """Alternative end-to-end membership pipeline using MST + Mahalanobis + GMM.

    This class implements the multi-stage workflow supplied by the science
    team that chains a robustly scaled minimum-spanning-tree selection with a
    Mahalanobis-distance clip and a smartly initialised Gaussian Mixture
    Model. It is intentionally lightweight and uses simple ``print`` logging so
    it can be dropped into exploratory notebooks.
    """

    def __init__(self, df: pd.DataFrame, center_params: dict) -> None:
        self.raw_data = df.copy()
        self.data = df.copy()
        self.center = center_params
        self.stats: dict[str, int] = {"init": len(df)}

    # ------------------------------------------------------------------
    @staticmethod
    def _riello_correction(bp_rp: pd.Series, excess: pd.Series) -> np.ndarray:
        """Riello et al. (2021) flux-excess correction."""

        bp_rp_arr = np.array(bp_rp, dtype=float)
        excess_arr = np.array(excess, dtype=float)
        correction = np.zeros_like(bp_rp_arr)
        valid = ~np.isnan(bp_rp_arr)

        mask_blue = valid & (bp_rp_arr < 0.5)
        mask_green = valid & (bp_rp_arr >= 0.5) & (bp_rp_arr < 4.0)
        mask_red = valid & (bp_rp_arr > 4.0)

        correction[mask_blue] = (
            1.154360
            + 0.033772 * bp_rp_arr[mask_blue]
            + 0.032277 * (bp_rp_arr[mask_blue] ** 2)
        )
        correction[mask_green] = (
            1.162004
            + 0.011464 * bp_rp_arr[mask_green]
            + 0.049255 * (bp_rp_arr[mask_green] ** 2)
            - 0.005879 * (bp_rp_arr[mask_green] ** 3)
        )
        correction[mask_red] = 1.057572 + 0.140537 * bp_rp_arr[mask_red]
        return excess_arr - correction

    @staticmethod
    def _sigma_cstar(g_mag: pd.Series) -> np.ndarray:
        """Intrinsic scatter of the C* diagnostic (Riello et al. 2021, Eq. 18)."""

        g_mag_arr = np.array(g_mag, dtype=float)
        return 0.0059898 + 8.817481e-12 * (g_mag_arr ** 7.618399)

    # ------------------------------------------------------------------
    def preprocess_data(self) -> "ScientificClusterPipeline":
        """Apply zero-point, photometric, and astrometric quality cuts."""

        print(f"1. Preprocessing... Input: {len(self.data)}")

        cols_zpt = [
            "phot_g_mean_mag",
            "ecl_lat",
            "nu_eff_used_in_astrometry",
            "pseudocolour",
            "astrometric_params_solved",
        ]
        if _HAS_ZPT and all(c in self.data.columns for c in cols_zpt):
            self.data["zpt"] = zpt.get_zpt(
                phot_g_mean_mag=self.data["phot_g_mean_mag"].values,
                ecl_lat=self.data["ecl_lat"].values,
                nu_eff_used_in_astrometry=self.data["nu_eff_used_in_astrometry"].values,
                pseudocolour=self.data["pseudocolour"].values,
                astrometric_params_solved=self.data["astrometric_params_solved"].values,
            )
            self.data["parallax_correction"] = self.data["parallax"] - self.data["zpt"]
        else:
            self.data["parallax_correction"] = self.data["parallax"]

        if "bp_rp" in self.data.columns:
            self.data["C_star"] = self._riello_correction(
                self.data["bp_rp"], self.data["phot_bp_rp_excess_factor"]
            )
            self.data["sigma_C_star"] = self._sigma_cstar(self.data["phot_g_mean_mag"])
            mask_phot = np.abs(self.data["C_star"]) < 5 * self.data["sigma_C_star"]
        else:
            mask_phot = np.ones(len(self.data), dtype=bool)

        mask_qual = (self.data["parallax"] > 0) & (
            (self.data["parallax"] / self.data["parallax_error"]) >= 5.0
        )

        self.data = self.data[mask_phot & mask_qual].dropna(
            subset=["ra", "dec", "parallax", "pmra", "pmdec"]
        )
        self.stats["cleaned"] = len(self.data)
        return self

    def coarse_filter(self, pm_sigma: float = 9.0, plx_sigma: float = 10.0) -> "ScientificClusterPipeline":
        """Apply coarse proper motion and parallax cuts around a supplied centre."""

        if "cleaned" not in self.stats:
            self.preprocess_data()

        pmra_err = self.center.get("pmra_error", 0.1)
        pmdec_err = self.center.get("pmdec_error", 0.1)
        plx_err = self.center.get("parallax_error", 0.1)

        pm_tol = pm_sigma * np.sqrt((pmra_err**2 + pmdec_err**2) / 2)
        plx_tol = plx_sigma * plx_err

        mask_pm = np.sqrt(
            (self.data["pmra"] - self.center["pmra"]) ** 2
            + (self.data["pmdec"] - self.center["pmdec"]) ** 2
        ) <= pm_tol
        mask_plx = np.abs(self.data["parallax"] - self.center["parallax"]) <= plx_tol

        self.data = self.data[mask_pm & mask_plx]
        self.stats["coarse"] = len(self.data)
        print(f"2. Coarse Filter: Count {len(self.data)}")
        return self

    def apply_mst(self, cut_sigma: float = 3.0) -> "ScientificClusterPipeline":
        """Minimum-spanning-tree selection with robust scaling."""

        if "coarse" not in self.stats:
            self.coarse_filter()

        df_mst = self.data.copy().reset_index(drop=True)
        scale_params = ["ra", "dec", "parallax_correction"]

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(df_mst[scale_params])

        dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
        mst_matrix = minimum_spanning_tree(dist_matrix)

        edges = mst_matrix.data
        mu = np.mean(edges)
        sigma = np.std(edges)
        threshold = mu + (cut_sigma * sigma)

        mst_dense = mst_matrix.toarray()
        mst_dense[mst_dense > threshold] = 0
        n_components, labels = connected_components(mst_dense, directed=False)

        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(counts) == 0:
            return self
        main_label = unique_labels[np.argmax(counts)]

        self.data = df_mst[labels == main_label].copy()
        self.stats["mst"] = len(self.data)
        print(f"3. MST (Robust): Threshold {threshold:.3f}, Count {len(self.data)}")
        return self

    def apply_mahalanobis_clipping(self, confidence: float = 0.997) -> "ScientificClusterPipeline":
        """Outlier rejection using a robust Mahalanobis distance cut."""

        if "mst" not in self.stats:
            self.apply_mst()

        features = ["ra", "dec", "parallax_correction", "pmra", "pmdec"]
        X = self.data[features].values

        try:
            robust_cov = MinCovDet().fit(X)
            m_dist = robust_cov.mahalanobis(X)

            threshold = np.sqrt(chi2.ppf(confidence, df=len(features)))

            self.data = self.data[m_dist < threshold]
            print(f"4. Mahalanobis Clip: Threshold {threshold:.2f}, Count {len(self.data)}")
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(
                "   ⚠️ Mahalanobis fitting failed (likely too few stars). Falling back "
                f"to Sigma Clip. Error: {exc}"
            )
            self.apply_sigma_clipping()

        self.stats["mahalanobis"] = len(self.data)
        return self

    def apply_sigma_clipping(self, sigma: float = 3.0) -> "ScientificClusterPipeline":
        """Fallback multivariate sigma clipping."""

        cols = ["ra", "dec", "parallax_correction", "pmra", "pmdec"]
        z = np.abs((self.data[cols] - self.data[cols].mean()) / self.data[cols].std())
        self.data = self.data[(z < sigma).all(axis=1)]
        return self

    def apply_gmm(self, prob_threshold: float = 0.8) -> pd.DataFrame:
        """Two-component GMM with smart initialisation for cluster vs. field."""

        if "mahalanobis" not in self.stats:
            self.apply_mahalanobis_clipping()

        features = ["ra", "dec", "parallax_correction", "pmra", "pmdec"]
        X = self.data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cluster_mean = np.mean(X_scaled, axis=0)
        field_offset = np.std(X_scaled, axis=0)

        means_init = np.vstack([cluster_mean, cluster_mean + field_offset])

        gmm = GaussianMixture(
            n_components=2,
            means_init=means_init,
            weights_init=[0.6, 0.4],
            random_state=42,
        )
        gmm.fit(X_scaled)

        probs = gmm.predict_proba(X_scaled)

        det_covs = [np.linalg.det(cov) for cov in gmm.covariances_]
        cluster_label = int(np.argmin(det_covs))

        self.data["gmm_prob"] = probs[:, cluster_label]
        self.data["cluster_pred_num"] = (gmm.predict(X_scaled) == cluster_label).astype(int)

        final_members = self.data[
            (self.data["cluster_pred_num"] == 1) & (self.data["gmm_prob"] >= prob_threshold)
        ].copy()

        self.stats["final"] = len(final_members)
        print(f"5. GMM (Smart Init): Members {len(final_members)} (P>={prob_threshold})")
        return final_members


class ClusterPlots:
    """Visualization utilities for cluster membership products."""

    def __init__(self) -> None:
        if _HAS_SCIENCEPLOTS:
            plt.style.use(["science", "ieee", "no-latex"])

    def cmd(self, data: pd.DataFrame, hue: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Color-Magnitude Diagram (G vs. BP-RP)."""

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5), dpi=120)
        palette = {
            "Field": "#9e9e9e",
            "MST candidate": "#1f77b4",
            "High-prob member": "#d62728",
        }
        sns.scatterplot(
            data=data,
            x="bp_rp",
            y="phot_g_mean_mag",
            hue=hue,
            palette=palette if hue == "population" else None,
            s=10,
            ax=ax,
            linewidth=0,
        )
        ax.invert_yaxis()
        ax.set_xlabel(r"$G_{BP} - G_{RP}$ (mag)")
        ax.set_ylabel(r"$G$ (mag)")
        return ax

    def spatial(self, data: pd.DataFrame, hue: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Spatial distribution: RA vs Dec (RA inverted)."""

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5), dpi=120)
        palette = {
            "Field": "#9e9e9e",
            "MST candidate": "#1f77b4",
            "High-prob member": "#d62728",
        }
        sns.scatterplot(
            data=data,
            x="ra",
            y="dec",
            hue=hue,
            palette=palette if hue == "population" else None,
            s=10,
            ax=ax,
            linewidth=0,
        )
        ax.invert_xaxis()
        ax.set_xlabel(r"RA (deg)")
        ax.set_ylabel(r"Dec (deg)")
        return ax

    def vpd(self, data: pd.DataFrame, hue: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Vector Point Diagram: pmra vs pmdec."""

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5), dpi=120)
        palette = {
            "Field": "#9e9e9e",
            "MST candidate": "#1f77b4",
            "High-prob member": "#d62728",
        }
        sns.scatterplot(
            data=data,
            x="pmra",
            y="pmdec",
            hue=hue,
            palette=palette if hue == "population" else None,
            s=10,
            ax=ax,
            linewidth=0,
        )
        ax.set_xlabel(r"$\mu_\alpha\cos\delta$ (mas yr$^{-1}$)")
        ax.set_ylabel(r"$\mu_\delta$ (mas yr$^{-1}$)")
        return ax


__all__ = ["ClusterMembership", "ClusterPlots", "ScientificClusterPipeline"]

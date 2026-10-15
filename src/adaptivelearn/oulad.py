"""Load OULAD and build per-enrolment feature tables at a given cut-off day.

One row = one student's enrolment in one module presentation. Only information
available up to `cutoff` (days since the presentation start) is used for
features; targets come from the future:

    final_result       Distinction / Pass / Fail / Withdrawn
    future_mean_score  mean score of non-exam assessments due AFTER the cut-off
"""
import numpy as np
import pandas as pd

from .config import RAW_DIR

CONTENT = {"oucontent", "resource", "subpage", "page", "url", "folder", "glossary",
           "dualpane", "sharedsubpage"}
FORUM = {"forumng", "oucollaborate", "ouelluminate", "ouwiki"}
QUIZ = {"quiz", "externalquiz", "questionnaire", "repeatactivity", "htmlactivity", "dataplus"}
GROUPS = ["homepage", "content", "forum", "quiz", "other"]


def _group(activity):
    if activity == "homepage":
        return 0
    if activity in CONTENT:
        return 1
    if activity in FORUM:
        return 2
    if activity in QUIZ:
        return 3
    return 4


def make_key(module, presentation, student, module_codes, pres_codes):
    """Single int64 key for an enrolment (fast group-bys on 10M rows)."""
    return ((module.map(module_codes).astype("int64") * len(pres_codes)
             + presentation.map(pres_codes).astype("int64")) * 10_000_000
            + student.astype("int64"))


class Oulad:
    """Raw OULAD tables plus the pre-aggregated click tables."""

    def __init__(self, raw_dir=RAW_DIR):
        r = raw_dir
        self.info = pd.read_csv(r / "studentInfo.csv")
        self.reg = pd.read_csv(r / "studentRegistration.csv")
        self.assess = pd.read_csv(r / "assessments.csv")
        self.stu_assess = pd.read_csv(r / "studentAssessment.csv")
        self.vle = pd.read_csv(r / "vle.csv")
        self.courses = pd.read_csv(r / "courses.csv")

        self.module_codes = {m: i for i, m in enumerate(sorted(self.info.code_module.unique()))}
        self.pres_codes = {p: i for i, p in enumerate(sorted(self.info.code_presentation.unique()))}
        self._key = lambda df: make_key(df.code_module, df.code_presentation, df.id_student,
                                        self.module_codes, self.pres_codes)

        sv = pd.read_csv(r / "studentVle.csv",
                         dtype={"id_site": "int32", "id_student": "int32",
                                "date": "int16", "sum_click": "int32"})
        site_group = self.vle.set_index("id_site")["activity_type"].map(_group).astype("int8")
        sv["grp"] = sv["id_site"].map(site_group).fillna(4).astype("int8")
        sv["key"] = self._key(sv)
        self.daily = (sv.groupby(["key", "date", "grp"], sort=False)["sum_click"].sum()
                      .rename("clicks").reset_index())
        self.visits = (sv.groupby(["key", "id_site"], sort=False)
                       .agg(first_date=("date", "min"), clicks=("sum_click", "sum")).reset_index())
        del sv

    # ------------------------------------------------------------- features
    def enrolments(self):
        e = self.info.merge(self.reg, on=["code_module", "code_presentation", "id_student"],
                            how="left")
        e = e.merge(self.courses, on=["code_module", "code_presentation"], how="left")
        e["key"] = self._key(e)
        e["season"] = e["code_presentation"].str[-1]
        e["imd_band"] = e["imd_band"].fillna("Unknown")
        return e

    def build(self, cutoff):
        """Feature + target table for students still enrolled at `cutoff`."""
        e = self.enrolments()
        e = e[~(e["date_unregistration"] <= cutoff)].copy()     # withdrew before cut-off
        e = e.set_index("key", drop=False)

        d = self.daily[self.daily["date"] <= cutoff]
        g = d.groupby("key")
        f = pd.DataFrame({
            "total_clicks": g["clicks"].sum(),
            "active_days": g["date"].nunique(),
            "last_active": g["date"].max(),
        })
        f["clicks_before_start"] = d[d["date"] < 0].groupby("key")["clicks"].sum()
        f["clicks_last_14d"] = d[d["date"] > cutoff - 14].groupby("key")["clicks"].sum()
        f["clicks_prev_14d"] = d[(d["date"] > cutoff - 28) & (d["date"] <= cutoff - 14)] \
            .groupby("key")["clicks"].sum()
        share = d.groupby(["key", "grp"])["clicks"].sum().unstack(fill_value=0)
        share = share.reindex(columns=range(5), fill_value=0)
        share = share.div(share.sum(axis=1).replace(0, np.nan), axis=0)
        for i, name in enumerate(GROUPS):
            f[f"share_{name}"] = share[i]
        f["n_sites"] = self.visits[self.visits["first_date"] <= cutoff].groupby("key").size()

        e = e.join(f)
        zero_cols = ["total_clicks", "active_days", "n_sites", "clicks_before_start",
                     "clicks_last_14d", "clicks_prev_14d"]
        e[zero_cols] = e[zero_cols].fillna(0)
        e["clicks_per_active_day"] = e["total_clicks"] / e["active_days"].replace(0, np.nan)
        e["clicks_per_active_day"] = e["clicks_per_active_day"].fillna(0)
        e["click_trend"] = (e["clicks_last_14d"] - e["clicks_prev_14d"]) / (e["clicks_prev_14d"] + 10)
        e["days_since_last_active"] = (cutoff - e["last_active"]).fillna(cutoff + 25)
        e = e.drop(columns="last_active")
        # no clicks at all -> shares are undefined; "other" = 1 would be a lie, leave NaN for imputer
        # ------------------------------------------------------------ assessments
        a = self.assess[self.assess["assessment_type"] != "Exam"].copy()
        sa = self.stu_assess.merge(a, on="id_assessment", how="inner")
        sa["key"] = self._key(sa)
        sa["late_days"] = sa["date_submitted"] - sa["date"]

        due = a[a["date"] <= cutoff].groupby(["code_module", "code_presentation"]).size() \
            .rename("n_assessments_due").reset_index()
        e = e.merge(due, on=["code_module", "code_presentation"], how="left").set_index("key", drop=False)
        e["n_assessments_due"] = e["n_assessments_due"].fillna(0)

        early = sa[(sa["date_submitted"] <= cutoff) & (sa["date"] <= cutoff)]
        ge = early.groupby("key")
        af = pd.DataFrame({
            "n_submitted": ge.size(),
            "mean_score_early": ge["score"].mean(),
            "min_score_early": ge["score"].min(),
            "n_late": ge["late_days"].apply(lambda s: int((s > 0).sum())),
            "mean_days_early": ge["late_days"].mean().mul(-1),
            "n_failed_early": ge["score"].apply(lambda s: int((s < 40).sum())),
        })
        e = e.join(af)
        e["n_submitted"] = e["n_submitted"].fillna(0)
        for c in ("n_late", "n_failed_early"):
            e[c] = e[c].fillna(0)
        e["submission_rate"] = np.where(e["n_assessments_due"] > 0,
                                        e["n_submitted"] / e["n_assessments_due"].replace(0, np.nan), np.nan)
        e["submission_rate"] = e["submission_rate"].clip(upper=1.0)

        fut = sa[sa["date"] > cutoff].groupby("key")["score"].mean().rename("future_mean_score")
        e = e.join(fut)
        e["registered_before_start"] = (e["date_registration"] < 0).astype(int)
        return e.reset_index(drop=True)

    # ------------------------------------------------------- recommender data
    def site_catalogue(self):
        v = self.vle[["id_site", "code_module", "code_presentation", "activity_type", "week_from"]]
        return v.copy()

    def upcoming_assessments(self, module, presentation, after_day):
        a = self.assess[(self.assess.code_module == module)
                        & (self.assess.code_presentation == presentation)
                        & (self.assess.date > after_day)]
        return a.sort_values("date")[["id_assessment", "assessment_type", "date", "weight"]]

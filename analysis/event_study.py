"""Event study — does uptime causally move share?

PRE-COMMITMENT (written before running on any data):
  If average within-(provider, model) share drops by ≥3pp in the 1-3 days
  following an uptime event (uptime_1d_pct falling >3pp below the provider's
  own 7-day rolling median AND below 95%), uptime causally moves share.
  Anything weaker than that (drop <1pp, or only on one provider, or only
  for one specific model) is suggestive but not conclusive — report as such.

Why this matters: cross-sectional identification on our 4-provider panel
failed because provider identity is collinear with everything else (R²=0.067
on Experiment A v1). Within-provider over time, identity is a constant by
construction — so any share response we see following a within-provider
uptime change is causally cleaner than any regression coefficient.

This is the standard interrupted-time-series / event-study fix when
cross-sections are confounded. We don't need a regression here; we need
to see whether share moves around discrete uptime events.

Method:
  1. For each (provider, model), compute the 7-day rolling median uptime
     (the provider's own baseline on this specific model).
  2. Define an "event" at date t as: uptime[t] < baseline[t] - 3pp AND uptime[t] < 95%.
  3. For each event, compute relative share at t-3 .. t+5, normalized to
     the pre-event mean (t-7 to t-2).
  4. Average across all events; check the 1-3 day post window.

Bonus case study: the 2026-05-17 DekaLLM-wide incident, where token share
dropped across all 5 DekaLLM models simultaneously. Surface this even if
the formal event window aggregate is thin.

Outputs:
  analysis/out/event_study.md            human-readable findings
  analysis/out/event_study.csv           event-level table for inspection
  analysis/out/event_study_response.png  averaged response curve

Caveats:
  Uptime data is only ~15 days deep, so the number of events is small (this
  is the constraint, not the method). With more uptime history (3+ months)
  this becomes a much stronger test. Treat current results as directional.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "out" / "share_panel_filled.csv"
OUT = ROOT / "out"

EVENT_DROP_THRESHOLD_PP = 3.0     # uptime must drop ≥3pp below baseline
EVENT_ABSOLUTE_THRESHOLD = 95.0   # AND uptime must be < 95% on the event day
BASELINE_WINDOW = 7               # days for rolling-median uptime baseline
PRE_WINDOW = (-7, -2)             # days relative to event for pre-event share baseline
POST_DAYS = list(range(-3, 6))    # days relative to event to plot share response


def main() -> None:
    if not PANEL.exists():
        raise SystemExit(f"Panel not found: {PANEL}. Run dump_prometheus_panel.py first.")

    df = pd.read_csv(PANEL, parse_dates=["date"])
    df = df.sort_values(["provider", "model_id", "date"]).reset_index(drop=True)

    # Restrict to rows with both share and uptime (the event signal needs both)
    df = df.dropna(subset=["token_share", "uptime_1d_pct"]).copy()
    if df.empty:
        raise SystemExit("No rows with both share and uptime — re-run panel dump.")

    print(f"Loaded {len(df):,} (provider, model, date) rows with share + uptime")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Providers : {sorted(df['provider'].unique())}")
    print()

    # ---- 1. Per-(provider, model) rolling-median uptime baseline ----
    df["uptime_baseline"] = (
        df.groupby(["provider", "model_id"])["uptime_1d_pct"]
        .transform(lambda s: s.shift(1).rolling(BASELINE_WINDOW, min_periods=3).median())
    )
    df["uptime_drop_pp"] = df["uptime_baseline"] - df["uptime_1d_pct"]

    # ---- 2. Identify events ----
    df["is_event"] = (
        (df["uptime_drop_pp"] >= EVENT_DROP_THRESHOLD_PP)
        & (df["uptime_1d_pct"] < EVENT_ABSOLUTE_THRESHOLD)
        & (df["uptime_baseline"].notna())
    )
    events = df[df["is_event"]].copy()
    print(f"Detected {len(events)} uptime events "
          f"(uptime drop ≥{EVENT_DROP_THRESHOLD_PP}pp AND uptime <{EVENT_ABSOLUTE_THRESHOLD}%)")
    if events.empty:
        print("\nNo qualifying events under the strict definition.")
        print("Relaxing: include any day where uptime drops ≥1pp from its 7-day median.")
        df["is_event"] = (
            (df["uptime_drop_pp"] >= 1.0) & (df["uptime_baseline"].notna())
        )
        events = df[df["is_event"]].copy()
        print(f"  Relaxed events: {len(events)}")

    # ---- 3. For each event, gather share at offset days ----
    df["share_t"] = df["token_share"]
    rows = []
    for _, ev in events.iterrows():
        prov = ev["provider"]
        model = ev["model_id"]
        ev_date = ev["date"]

        own = df[(df["provider"] == prov) & (df["model_id"] == model)].set_index("date")

        # Pre-event baseline = mean share over PRE_WINDOW
        pre_start = ev_date + pd.Timedelta(days=PRE_WINDOW[0])
        pre_end = ev_date + pd.Timedelta(days=PRE_WINDOW[1])
        pre_mask = (own.index >= pre_start) & (own.index <= pre_end)
        pre_shares = own.loc[pre_mask, "share_t"].dropna()
        if len(pre_shares) < 2:
            continue
        baseline_share = pre_shares.mean()
        if baseline_share <= 0:
            continue

        for k in POST_DAYS:
            day = ev_date + pd.Timedelta(days=k)
            if day in own.index:
                share_at = float(own.loc[day, "share_t"])
                rows.append({
                    "event_provider": prov,
                    "event_model_id": model,
                    "event_date": ev_date,
                    "event_uptime": ev["uptime_1d_pct"],
                    "event_uptime_drop_pp": ev["uptime_drop_pp"],
                    "offset_days": k,
                    "share_at_offset": share_at,
                    "baseline_share": baseline_share,
                    "relative_share": share_at / baseline_share,
                    "share_delta_pp": (share_at - baseline_share) * 100,
                })

    rt = pd.DataFrame(rows)
    if rt.empty:
        raise SystemExit("No events with enough pre/post coverage — uptime history too short.")
    rt.to_csv(OUT / "event_study.csv", index=False)

    # ---- 4. Aggregate response curve ----
    response = rt.groupby("offset_days").agg(
        n=("share_delta_pp", "count"),
        mean_delta_pp=("share_delta_pp", "mean"),
        median_delta_pp=("share_delta_pp", "median"),
        std_delta_pp=("share_delta_pp", "std"),
    ).round(3)

    print("\n=== Average share response around uptime events ===")
    print("(share_delta_pp = actual share − pre-event mean share, in percentage points)")
    print()
    print(response.to_string())
    print()

    # Mentor's pre-committed test
    post_1_3 = rt[rt["offset_days"].isin([1, 2, 3])]
    if not post_1_3.empty:
        avg_post = post_1_3["share_delta_pp"].mean()
        med_post = post_1_3["share_delta_pp"].median()
        n_post = len(post_1_3)
        # one-sample t-test against 0 (no effect)
        from scipy import stats as scistats
        t_stat, p_val = scistats.ttest_1samp(post_1_3["share_delta_pp"].dropna(), 0.0)
        print(f"=== Pre-committed test: post-event 1-3 day window ===")
        print(f"  n = {n_post}, mean Δshare = {avg_post:+.2f} pp, "
              f"median = {med_post:+.2f} pp")
        print(f"  one-sample t vs 0: t = {t_stat:+.3f}, p = {p_val:.4f}")
        print()
        if avg_post <= -3.0 and p_val < 0.05:
            verdict = "STRONG: avg drop ≥3pp AND p<0.05 — uptime causally moves share"
        elif avg_post <= -1.0:
            verdict = "DIRECTIONAL: avg drop 1-3pp — suggestive, not yet conclusive"
        else:
            verdict = "WEAK: avg drop <1pp — no clear within-provider uptime effect in this window"
        print(f"  VERDICT: {verdict}")
    else:
        print("No post-event observations in window.")

    # ---- 5. May 17 case study (the DekaLLM-wide incident) ----
    print("\n=== Case study: 2026-05-17 DekaLLM incident ===")
    may17 = pd.Timestamp("2026-05-17")
    may16 = pd.Timestamp("2026-05-16")
    may18 = pd.Timestamp("2026-05-18")
    dek_panel = df[df["provider"] == "dekallm"].set_index(["date", "model_id"])
    if (may17, ) in [(idx[0],) for idx in dek_panel.index]:
        for model_id in sorted(dek_panel.index.get_level_values("model_id").unique()):
            try:
                s16 = float(dek_panel.loc[(may16, model_id), "token_share"])
                s17 = float(dek_panel.loc[(may17, model_id), "token_share"])
                s18 = float(dek_panel.loc[(may18, model_id), "token_share"])
                u17 = float(dek_panel.loc[(may17, model_id), "uptime_1d_pct"])
                print(f"  {model_id:<55} 05-16: {s16*100:>5.2f}%  "
                      f"05-17: {s17*100:>5.2f}% (uptime {u17:.2f}%)  "
                      f"05-18: {s18*100:>5.2f}%  drop: {(s17-s16)*100:+.2f}pp")
            except KeyError:
                continue

    # ---- 6. Plot ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        x = response.index.to_numpy()
        y = response["mean_delta_pp"].to_numpy()
        se = response["std_delta_pp"].to_numpy() / np.sqrt(response["n"].to_numpy())
        ax.errorbar(x, y, yerr=1.96 * se, marker="o", capsize=4, linewidth=2,
                    label="mean ± 95% CI")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
                   label="event day")
        ax.set_xlabel("days from uptime event")
        ax.set_ylabel("share Δ vs pre-event baseline (pp)")
        ax.set_title(f"Average share response around {len(rt['event_provider'].unique())}-event uptime drops "
                     f"({len(rt)} obs)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = OUT / "event_study_response.png"
        fig.savefig(png_path, dpi=120)
        print(f"\nSaved plot:  {png_path}")
    except ImportError:
        pass

    # ---- 7. Markdown report ----
    md = []
    md.append("# Event study — does uptime move share?\n")
    md.append(f"Generated from {len(rt['event_provider'].unique())} "
              f"distinct (provider, model) events, {len(rt)} share observations.\n")
    md.append(f"\n## Pre-commitment\n{__doc__.split('PRE-COMMITMENT')[1].split('Why this matters')[0].strip()}\n")
    md.append("\n## Response curve\n")
    md.append("share_delta_pp = (share on day t+k) − (mean share over pre-event window).\n\n")
    md.append("| offset_days | n | mean Δpp | median Δpp | std Δpp |")
    md.append("|---:|---:|---:|---:|---:|")
    for offs, row in response.iterrows():
        md.append(f"| {offs} | {int(row['n'])} | {row['mean_delta_pp']:+.2f} | "
                  f"{row['median_delta_pp']:+.2f} | {row['std_delta_pp']:.2f} |")
    md.append(f"\n## Verdict\n\n{verdict}\n")
    with open(OUT / "event_study.md", "w") as f:
        f.write("\n".join(md))
    print(f"Saved md:    {OUT / 'event_study.md'}")
    print(f"Saved csv:   {OUT / 'event_study.csv'}")


if __name__ == "__main__":
    main()

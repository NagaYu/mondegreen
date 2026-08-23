"""The two headline figures, plus supporting plots.

Figure 1 -- term recall vs glossary size
    The point of the whole project in one picture.  Baseline (B) stuffs the
    glossary into Whisper's ``initial_prompt`` and flattens the moment the
    glossary outgrows 244 tokens; Mondegreen indexes phonetically and keeps
    going.  The x-axis is log-scaled because the interesting behaviour is
    order-of-magnitude behaviour.

Figure 2 -- correction rate vs damage rate
    Every point is a gate threshold.  Up is more of the broken terms fixed;
    right is more of the already-correct text broken.  A post-corrector that
    cannot show you this curve is asking you to trust it.

Any figure built from simulated data is watermarked.  That is not decoration:
a plot that leaves this repo without its provenance attached is a plot that will
eventually be quoted as a measurement.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Condition -> (colour, marker, z-order).  Mondegreen conditions are drawn last
#: and heaviest.
STYLE: Dict[str, Dict[str, object]] = {
    "A": {"color": "#9aa0a6", "marker": "o", "ls": "--", "lw": 1.6, "z": 1},
    "B": {"color": "#d93025", "marker": "s", "ls": "-", "lw": 2.0, "z": 3},
    "C": {"color": "#f9ab00", "marker": "^", "ls": "-", "lw": 2.0, "z": 2},
    "D": {"color": "#1a73e8", "marker": "o", "ls": "-", "lw": 2.8, "z": 5},
    "E": {"color": "#12b5cb", "marker": "D", "ls": ":", "lw": 2.2, "z": 4},
}

SHORT_LABELS = {
    "A": "(A) raw Whisper",
    "B": "(B) Whisper prompt (244-token cap)",
    "C": "(C) cloud LLM post-processing",
    "D": "(D) Mondegreen",
    "E": "(E) Mondegreen, Q4_K_M",
}


#: Fonts that can render the Japanese axis labels, in order of preference.
#: Without one of these, matplotlib silently draws every kanji as a tofu box and
#: the figures ship unreadable.
_CJK_FONTS: Tuple[str, ...] = (
    "Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "Meiryo",
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic",
    "BIZ UDGothic", "Apple SD Gothic Neo", "Arial Unicode MS", "Source Han Sans JP",
)


def _setup_fonts(plt) -> Optional[str]:
    """Pick a CJK-capable font, or fall back to English-only labels.

    Claim: SUPPORT -- a figure whose axis labels are empty rectangles is not a
    result, and matplotlib will produce one without raising.
    """
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.family"] = ["sans-serif"]
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


#: Set once :func:`_require_mpl` runs; ``None`` means labels fall back to English.
CJK_FONT: Optional[str] = None


def _require_mpl():
    """Import matplotlib in headless mode and select a CJK-capable font.

        Claim: SUPPORT.
        """
    global CJK_FONT
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        CJK_FONT = _setup_fonts(plt)
        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for figures: pip install 'mondegreen[figures]'"
        ) from exc


def _label(ja: str, en: str) -> str:
    """Bilingual label when a CJK font exists, English-only when it does not.

    Claim: SUPPORT -- degrade to readable rather than to tofu.
    """
    return f"{ja}  /  {en}" if CJK_FONT else en


def _watermark(ax, results: Dict[str, object]) -> None:
    """Stamp SIMULATED across any plot not built from measurements.

    Claim: SUPPORT -- provenance has to travel with the picture, because the
    picture is what gets screenshotted.
    """
    provs = {str(r.get("provenance")) for r in results.get("rows", [])}  # type: ignore[union-attr]
    if provs and provs <= {"measured"}:
        return
    ax.text(
        0.5, 0.5, "SIMULATED",
        transform=ax.transAxes, fontsize=44, color="#000000", alpha=0.07,
        ha="center", va="center", rotation=24, zorder=0, fontweight="bold",
    )


def _series(
    results: Dict[str, object], metric: str, sweep: str = "coverage"
) -> Dict[str, List[Tuple[int, float]]]:
    """Extract ``{condition: [(glossary_size, value)]}`` for one metric and sweep.

        Claim: SUPPORT.
        """
    out: Dict[str, List[Tuple[int, float]]] = {}
    for row in results.get("rows", []):  # type: ignore[union-attr]
        if str(row.get("sweep", "coverage")) != sweep:
            continue
        cond = str(row["condition"])
        size = int(row["glossary_size"])
        m = row["metrics"]
        if metric == "term_recall":
            v = float(m["term_recall"]["recall"])
        elif metric.startswith("damage_"):
            v = float(m["damage"][metric])
        elif metric == "hallucination_removal":
            v = float(m.get("hallucination", {}).get("removal_rate", 0.0))
        else:
            v = float(m[metric])
        out.setdefault(cond, []).append((size, v))
    return {k: sorted(v) for k, v in out.items()}


def figure_recall_vs_glossary_size(
    results: Dict[str, object],
    out_path: str = "figures/term_recall_vs_glossary_size.png",
    title: Optional[str] = None,
) -> str:
    """Figure 1: the headline.  (B) plateaus at the prompt ceiling; (D) does not.

    Claim: UNBOUNDED-VOCAB -- this figure *is* the claim.
    """
    plt = _require_mpl()
    title = title or _label("用語の再現率 vs 用語集サイズ", "Term recall vs glossary size")
    series = _series(results, "term_recall")
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=160)
    _watermark(ax, results)

    # (E) is drawn first and thicker so that when it coincides exactly with (D)
    # -- which it does whenever no quantised checkpoint was supplied -- it shows
    # as a halo rather than vanishing underneath.
    identical = series.get("D") and series.get("E") and series["D"] == series["E"]
    for cond in ("A", "B", "C", "E", "D"):
        pts = series.get(cond)
        if not pts:
            continue
        st = STYLE[cond]
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        label = SHORT_LABELS[cond]
        lw, ms = st["lw"], 6
        if cond == "E" and identical:
            label += " — identical to (D) here"
            lw, ms = 6.5, 10
        ax.plot(xs, ys, label=label, color=st["color"], marker=st["marker"],
                linestyle=st["ls"], linewidth=lw, markersize=ms,
                zorder=st["z"], alpha=0.55 if (cond == "E" and identical) else 1.0)

    # Mark where the glossary stops fitting in Whisper's prompt.
    ceiling = _prompt_ceiling(results)
    if ceiling:
        ax.axvline(ceiling, color="#d93025", alpha=0.55, linestyle=":", linewidth=1.6, zorder=1)
        ax.annotate(
            f"Whisper の initial_prompt はここで満杯\n"
            f"≈{ceiling} terms = 244 tokens\n"
            f"→ (B) はこれより右を一切保持できない" if CJK_FONT else
            f"Whisper initial_prompt is full here\n"
            f"≈{ceiling} terms = 244 tokens\n"
            f"→ (B) can never hold anything to the right",
            xy=(ceiling, 29), xytext=(0.055, 0.80), textcoords="axes fraction",
            fontsize=8.4, color="#d93025", va="top",
            arrowprops=dict(arrowstyle="->", color="#d93025", alpha=0.7, lw=1.1),
        )

    ax.set_xscale("log")
    ax.set_xlabel(_label("用語集サイズ", "Glossary size") + " (terms) — log scale")
    ax.set_ylabel(_label("用語の再現率", "Term recall") + " (%)")
    ax.set_title(title, fontsize=11.5, pad=12)
    ax.set_ylim(0, 100)
    caption = (
        "用語集が大きいほど、実際に話される語をより多く含む。(B) だけがその恩恵を受けられない。"
        if CJK_FONT else
        "A larger glossary covers more of what is actually spoken. (B) is the only "
        "method that cannot benefit."
    )
    ax.text(0.5, -0.16, caption, transform=ax.transAxes, fontsize=8.6,
            color="#5f6368", ha="center", va="top")
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
    ax.legend(loc="lower left", fontsize=8.4, framealpha=0.94, borderpad=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_path)


def figure_correction_vs_damage(
    results: Dict[str, object],
    out_path: str = "figures/correction_vs_damage.png",
    title: Optional[str] = None,
) -> str:
    """Figure 2: the safety frontier.

    One curve per value of the hard bound tau; along each curve, the gate
    threshold varies.  Reading it: moving right along a curve is the gate being
    relaxed, and moving to a higher curve is the *bound* being loosened.  Almost
    all of the safety comes from the bound -- the tight-tau curves hug the y-axis
    -- which is the central design claim of the project stated as a picture.

    Claim: LOW-DAMAGE.
    """
    plt = _require_mpl()
    title = title or _label("訂正率 vs 破壊率", "Correction rate vs damage rate")
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=160)
    _watermark(ax, results)

    grid = results.get("eval_grid") or {}
    fallback = results.get("eval_sweep") or results.get("gate", {}).get("train_sweep") or []
    if not grid and not fallback:
        ax.text(0.5, 0.5, "no sweep data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out_path)

    if grid:
        taus = sorted(grid, key=float)
        cmap = plt.get_cmap("viridis")
        for k, tau in enumerate(taus):
            pts = sorted(grid[tau], key=lambda p: (p["damage_rate"], p["correction_rate"]))
            xs = [p["damage_rate"] * 100 for p in pts]
            ys = [p["correction_rate"] * 100 for p in pts]
            color = cmap(0.12 + 0.76 * k / max(1, len(taus) - 1))
            ax.plot(xs, ys, color=color, linewidth=2.3, marker="o", markersize=3.4,
                    label=f"base = {tau}", zorder=3 + k)
    else:
        pts = sorted(fallback, key=lambda p: p["damage_rate"])
        ax.plot([p["damage_rate"] * 100 for p in pts],
                [p["correction_rate"] * 100 for p in pts],
                color="#1a73e8", linewidth=2.6, zorder=3)

    shipped = results.get("gate", {}).get("chosen_threshold")  # type: ignore[union-attr]
    tau_cfg = results.get("config", {}).get("tau")             # type: ignore[union-attr]
    ref = (grid.get("0.2") or grid.get("0.4")) if grid else fallback
    if ref and shipped is not None:
        near = min(ref, key=lambda p: abs(p["threshold"] - float(shipped)))
        ax.scatter([near["damage_rate"] * 100], [near["correction_rate"] * 100],
                   s=170, facecolor="none", edgecolor="#d93025", linewidth=2.3, zorder=9)
        ax.annotate(
            f"shipped default\nτ={float(tau_cfg):.2f}, gate={near['threshold']:.2f}\n"
            f"correct {near['correction_rate']:.0%} / damage {near['damage_rate']:.2%}",
            xy=(near["damage_rate"] * 100, near["correction_rate"] * 100),
            xytext=(0.30, 0.90), textcoords="axes fraction", fontsize=8.5, color="#d93025",
            va="top", arrowprops=dict(arrowstyle="->", color="#d93025", alpha=0.75, lw=1.1),
        )

    budget = results.get("config", {}).get("max_damage_rate")  # type: ignore[union-attr]
    if budget:
        ax.axvline(float(budget) * 100, color="#5f6368", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.text(float(budget) * 100, 76.5, f" damage budget {float(budget):.1%}",
                fontsize=8.2, color="#5f6368", va="top")

    # Symlog: the whole argument lives between 0% and 1% damage, and a linear
    # axis stretched out to the reckless end would render it as a single stripe.
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.5)
    ax.set_xlim(left=-0.02)
    ax.xaxis.set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(
            lambda v, _: ("0" if v == 0 else (f"{v:g}" if v < 1 else f"{v:.0f}"))
        )
    )
    ax.set_xlabel(_label("破壊率", "Damage rate")
                  + " (%) — already-correct term occurrences broken  (symlog)")
    ax.set_ylabel(_label("訂正率", "Correction rate") + " (%) — broken term occurrences repaired")
    ax.set_title(title, fontsize=11.5, pad=12)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_ylim(0, 80)
    ax.legend(loc="lower right", fontsize=8.4,
              title="hard bound:  max_raw = base + 0.20·√mora",
              title_fontsize=8.0, framealpha=0.94)
    caption2 = (
        "曲線間の移動が「制約」の緩め方、曲線上の移動が「ゲート」の緩め方。安全性のほとんどは制約側から来る。"
        if CJK_FONT else
        "Between curves = loosening the constraint; along a curve = loosening the gate. "
        "Nearly all of the safety comes from the constraint."
    )
    ax.text(0.5, -0.17, caption2, transform=ax.transAxes, fontsize=8.6,
            color="#5f6368", ha="center", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_path)


def figure_cer_by_condition(
    results: Dict[str, object],
    out_path: str = "figures/cer_by_condition.png",
) -> str:
    """Supporting: CER and damage side by side, at the largest glossary size.

    Claim: LOW-DAMAGE -- CER alone can hide breakage, so it is never shown alone.
    """
    plt = _require_mpl()
    cov = [r for r in results.get("rows", []) if str(r.get("sweep", "coverage")) == "coverage"]
    sizes = sorted({int(r["glossary_size"]) for r in cov})
    if not sizes:
        raise ValueError("no rows in results")
    size = sizes[-1]
    rows = [r for r in cov if int(r["glossary_size"]) == size]
    rows.sort(key=lambda r: str(r["condition"]))

    conds = [str(r["condition"]) for r in rows]
    cers = [float(r["metrics"]["cer"]) * 100 for r in rows]
    dmg = [float(r["metrics"]["damage"]["damage_rate_chars"]) * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=160)
    _watermark(ax1, results)
    colors = [STYLE[c]["color"] for c in conds]
    ax1.bar(conds, cers, color=colors)
    ax1.set_title(f"CER (%) — glossary size {size}", fontsize=10.5)
    ax1.set_ylabel("CER (%)")
    for i, v in enumerate(cers):
        ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4)

    ax2.bar(conds, dmg, color=colors)
    ax2.set_title(_label("破壊率", "Damage rate") + " (%) — lower is better", fontsize=10.5)
    ax2.set_ylabel("damage rate (%)")
    for i, v in enumerate(dmg):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.4)
    for ax in (ax1, ax2):
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_path)


def figure_damage_vs_glossary_size(
    results: Dict[str, object],
    out_path: str = "figures/damage_vs_glossary_size.png",
) -> str:
    """Supporting: does damage grow as the glossary does?

    This uses the **distractor control** sweep, not the coverage sweep: the target
    terms are held fixed and the glossary is padded with unrelated vocabulary.
    That isolates the question exactly -- more terms means more phonetic
    neighbours and more chances to pick the wrong one, with no confounding change
    in what is being asked for.  If (D)'s damage climbed here, the headline figure
    would have been bought with breakage.

    Claim: LOW-DAMAGE + UNBOUNDED-VOCAB.
    """
    plt = _require_mpl()
    series = _series(results, "damage_rate_chars", sweep="distractor")
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    _watermark(ax, results)
    for cond in ("B", "C", "E", "D"):
        pts = series.get(cond)
        if not pts:
            continue
        st = STYLE[cond]
        ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], label=SHORT_LABELS[cond],
                color=st["color"], marker=st["marker"], linestyle=st["ls"],
                linewidth=st["lw"], markersize=6, zorder=st["z"])
    ax.set_xscale("log")
    ax.set_xlabel(_label("用語集サイズ", "Glossary size") + " (terms) — log scale")
    ax.set_ylabel(_label("破壊率", "Damage rate") + " (%)")
    ax.set_title(
        _label("破壊率 vs 用語集サイズ（対象語は固定・distractor のみ増加）",
               "Damage rate vs glossary size (targets fixed, distractors added)"),
        fontsize=10.5, pad=10,
    )
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
    ax.legend(fontsize=8.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_path)


def _prompt_ceiling(results: Dict[str, object]) -> Optional[int]:
    """How many terms fit in 244 tokens, read out of the (B) rows.

    Claim: UNBOUNDED-VOCAB.
    """
    for row in results.get("rows", []):  # type: ignore[union-attr]
        if str(row.get("condition")) == "B" and str(row.get("sweep", "coverage")) == "coverage":
            cap = (row.get("extra") or {}).get("prompt_capacity") or {}
            n = cap.get("terms_included")
            if n:
                return int(n)
    return None


def _save(fig, out_path: str) -> str:
    """Write PNG and a matching SVG next to it.

    Claim: SUPPORT -- the README embeds the PNG; the SVG is for anyone who wants
    to look closely.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    svg = os.path.splitext(out_path)[0] + ".svg"
    fig.savefig(svg, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


def make_all(results: Dict[str, object], out_dir: str = "figures") -> List[str]:
    """Generate every figure from one benchmark result dict.

    Claim: SUPPORT.
    """
    return [
        figure_recall_vs_glossary_size(results, os.path.join(out_dir, "term_recall_vs_glossary_size.png")),
        figure_correction_vs_damage(results, os.path.join(out_dir, "correction_vs_damage.png")),
        figure_cer_by_condition(results, os.path.join(out_dir, "cer_by_condition.png")),
        figure_damage_vs_glossary_size(results, os.path.join(out_dir, "damage_vs_glossary_size.png")),
    ]


def load_results(path: str) -> Dict[str, object]:
    """Read a benchmark JSON file.

    Claim: SUPPORT.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

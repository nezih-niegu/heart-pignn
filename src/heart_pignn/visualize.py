"""Validation visualizer: a real-time ECG monitor, plus a final summary figure.

Two outputs, because they answer different questions:

- **`*_realtime.gif`** replays a continuous strip from one record the way a bedside
  monitor would, beat by beat. Each beat lights up as the model classifies it, a
  side panel shows the regression head's reconstruction against the true
  waveform, and a conduction-graph panel animates the PIGNN's depolarization
  wave (`vm` per node, per graph step) for the beat under the cursor -- so you
  watch activation travel SA -> AV -> His -> Purkinje -> ventricles in sync with
  the trace. This is where you *see* failure modes a confusion matrix hides --
  for instance the model flipping between S and V on consecutive beats of the
  same morphology.
- **`*_summary.png`** is the static end state: the whole strip with every beat
  marked correct or wrong, the reconstruction overlay, and the metrics for that
  record.

Both run on beats the model never trained on. The strip is real signal from the
record, not a concatenation of windows, so RR timing is genuine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write files directly
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from .aami import CLASS_NAMES  # noqa: E402
from .data import MITBIHBeatDataset  # noqa: E402

# Colour-blind-safe palette (Okabe-Ito), because red/green correct-vs-wrong is
# unreadable for roughly 1 in 12 men.
CLASS_COLORS = {
    "N": "#0072B2", "S": "#E69F00", "V": "#D55E00", "F": "#CC79A7", "Q": "#009E73",
}
OK_COLOR, BAD_COLOR = "#009E73", "#D55E00"


@dataclass
class BeatPrediction:
    position: int          # sample index within the strip
    true_class: int
    pred_class: int
    confidence: float
    window: np.ndarray     # model input,  [T]
    target: np.ndarray     # regression target, [T]
    recon: np.ndarray      # regression output, [T]
    has_target: bool
    vm: np.ndarray | None = None  # PIGNN membrane potential per graph-step, [S, N]

    @property
    def correct(self) -> bool:
        return self.true_class == self.pred_class


def _record_with_class(dataset: MITBIHBeatDataset, focus_class: int) -> str | None:
    """Pick the split record holding the most beats of `focus_class`.

    Most MIT-BIH records are N-dominated, so a contiguous strip of a random
    record rarely contains the rarer classes. This finds a record that actually
    has the class, so the monitor has something to show.
    """
    best_rec, best_count = None, 0
    for rec in dataset.records:
        idx = dataset.beats_in_record(rec)
        count = int(np.sum(dataset.labels[idx] == focus_class))
        if count > best_count:
            best_rec, best_count = rec, count
    return best_rec


def _seek_window(labels: np.ndarray, n_beats: int, focus_class: int) -> tuple[int, int]:
    """Slide an n_beats window over the record and stop where it holds the most focus beats.

    Keeps the strip contiguous -- it relocates *where* on the record we look, it
    does not stitch non-adjacent beats together, so the RR timing stays genuine.
    """
    n = len(labels)
    if n <= n_beats:
        return 0, n
    hits = (labels == focus_class).astype(np.int32)
    window = int(hits[:n_beats].sum())
    best_start, best_hits = 0, window
    for start in range(1, n - n_beats + 1):
        window += int(hits[start + n_beats - 1]) - int(hits[start - 1])
        if window > best_hits:
            best_start, best_hits = start, window
    return best_start, best_start + n_beats


@torch.no_grad()
def collect_predictions(
    model,
    dataset: MITBIHBeatDataset,
    device,
    record: str,
    n_beats: int = 25,
    batch_size: int = 64,
    focus_class: int | None = None,
) -> tuple[np.ndarray, list[BeatPrediction], int]:
    """Run the model over consecutive beats of one record and return the raw strip too.

    With `focus_class` set, the window slides to the busiest stretch for that
    class instead of starting at beat 0; the strip stays contiguous either way.
    """
    model.eval()
    idx = dataset.beats_in_record(record)
    if len(idx) == 0:
        raise ValueError(f"record '{record}' has no beats in split '{dataset.split}'")

    if focus_class is not None:
        lo, hi = _seek_window(dataset.labels[idx], n_beats, focus_class)
        n_found = int(np.sum(dataset.labels[idx][lo:hi] == focus_class))
        print(
            f"  focusing on class {CLASS_NAMES[focus_class]}: "
            f"{n_found}/{hi - lo} beats in this window of record {record}"
        )
        if n_found == 0:
            print(f"  note: record {record} has no {CLASS_NAMES[focus_class]} beats; showing beat 0 on")
        idx = idx[lo:hi]
    else:
        idx = idx[:n_beats]

    signal, fs = dataset._load_signal(record, return_fs=True)
    channel = signal[:, dataset.cfg.channel]
    positions = dataset.sample_positions[idx]

    pre, post = dataset.cfg.pre_r, dataset.cfg.window_len
    strip_start = max(0, int(positions[0]) - pre - int(0.3 * fs))
    strip_end = min(len(channel), int(positions[-1]) + post + int(0.3 * fs))
    strip = channel[strip_start:strip_end]

    beats: list[BeatPrediction] = []
    for lo in tqdm(range(0, len(idx), batch_size), desc="predicting", unit="batch", leave=False):
        chunk = idx[lo : lo + batch_size]
        items = [dataset[int(i)] for i in chunk]
        batch = {k: torch.stack([it[k] for it in items]).to(device) for k in items[0]}
        out = model(batch["x"], batch["rule_vec"], batch["regime"])
        probs = torch.softmax(out["logits"].float(), dim=1).cpu().numpy()
        recon = (
            out["signal"].float().cpu().numpy()[..., 0]
            if "signal" in out
            else np.zeros((len(chunk), dataset.cfg.window_len), dtype=np.float32)
        )
        # vm is [B, S, N]: membrane potential at every node, at every graph step.
        # This is what lets the monitor animate the depolarization wave.
        vm = out["vm"].float().cpu().numpy() if "vm" in out else None
        for j, i in enumerate(chunk):
            beats.append(
                BeatPrediction(
                    position=int(dataset.sample_positions[int(i)]) - strip_start,
                    true_class=int(items[j]["y"]),
                    pred_class=int(probs[j].argmax()),
                    confidence=float(probs[j].max()),
                    window=items[j]["x"].numpy()[0],
                    target=items[j]["y_signal"].numpy()[:, 0],
                    recon=recon[j],
                    has_target=bool(float(items[j]["signal_mask"]) > 0),
                    vm=vm[j] if vm is not None else None,
                )
            )
    return strip, beats, fs


# ------------------------------------------------------------------- rendering


def _style_axis(ax, fs: int, title: str = "") -> None:
    """ECG-paper look: light grid, no top/right spines."""
    ax.set_facecolor("#FDF6F6")
    ax.grid(which="major", color="#E8B4B4", linewidth=0.6, alpha=0.8)
    ax.grid(which="minor", color="#F5DADA", linewidth=0.4, alpha=0.8)
    ax.minorticks_on()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")


def _init_conduction_graph(ax, spec):
    """Draw the static conduction graph and return the node scatter to update per frame.

    Node positions are the anatomical coordinates from the graph spec (SA at the
    top, ventricular nodes at the bottom), so the animation reads top-to-bottom
    the way real depolarization travels.
    """
    from matplotlib.collections import LineCollection

    coords = spec.coords.detach().cpu().numpy()
    xs, ys = coords[:, 0], coords[:, 1]
    ei = spec.edge_index.detach().cpu().numpy()

    segments = [[(xs[s], ys[s]), (xs[d], ys[d])] for s, d in zip(ei[0], ei[1], strict=False)]
    ax.add_collection(LineCollection(segments, colors="#D8D8D8", linewidths=0.6, zorder=1))

    scat = ax.scatter(
        xs, ys, c=np.zeros(len(xs)), cmap="inferno", vmin=0.0, vmax=1.0,
        s=70, zorder=2, edgecolors="#444444", linewidths=0.5,
    )
    # Label the landmarks of the conduction path so the wave is legible.
    for name in ("SA", "AV", "HIS"):
        if name in spec.node_names:
            i = spec.node_names.index(name)
            ax.annotate(name, (xs[i], ys[i]), fontsize=7, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points", color="#333333")
    pad_x = (xs.max() - xs.min()) * 0.15 + 0.05
    pad_y = (ys.max() - ys.min()) * 0.15 + 0.05
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("PIGNN conduction graph", fontsize=10, loc="left", fontweight="bold")
    return scat


def _graph_step_for(beat: BeatPrediction, cursor: int, fs: int, window_len: int) -> np.ndarray:
    """Map the scroll cursor's position within the beat to one graph step of vm.

    As the cursor moves across the beat, the returned activation advances through
    the PIGNN's internal graph steps -- so the wave propagates in sync with the
    strip. Values are min-max normalized within the beat so colour spans 0..1.
    """
    if beat.vm is None:
        return np.zeros(1)
    s = beat.vm.shape[0]
    frac = np.clip((cursor - beat.position) / max(window_len, 1), 0.0, 1.0)
    step = min(s - 1, int(frac * s))
    v = beat.vm
    lo, hi = float(v.min()), float(v.max())
    return (v[step] - lo) / (hi - lo + 1e-8)


def render_realtime(
    strip: np.ndarray,
    beats: list[BeatPrediction],
    fs: int,
    out_path: Path,
    record: str,
    fps: int = 20,
    seconds_visible: float = 4.0,
    speed: float = 1.0,
    graph_spec=None,
) -> Path:
    """Animate the strip as a scrolling monitor and write a GIF.

    When `graph_spec` is given and the beats carry `vm`, a conduction-graph panel
    animates the depolarization wave for the beat currently under the cursor.
    """
    window_samples = int(seconds_visible * fs)
    step = max(1, int(fs / fps * speed))
    n_frames = max(1, (len(strip) - window_samples) // step + 1)

    show_graph = graph_spec is not None and any(b.vm is not None for b in beats)

    # dpi kept modest: GIF size scales with frame area, and these get long.
    if show_graph:
        fig = plt.figure(figsize=(13.5, 7), dpi=95)
        gs = fig.add_gridspec(
            2, 3, height_ratios=[2, 1.25], width_ratios=[2, 1.05, 0.85],
            hspace=0.35, wspace=0.24,
        )
        ax_strip = fig.add_subplot(gs[0, :])
        ax_beat = fig.add_subplot(gs[1, 0])
        ax_graph = fig.add_subplot(gs[1, 1])
        ax_tally = fig.add_subplot(gs[1, 2])
        graph_scatter = _init_conduction_graph(ax_graph, graph_spec)
    else:
        fig = plt.figure(figsize=(12, 6.5), dpi=95)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.1], width_ratios=[2.2, 1], hspace=0.35, wspace=0.22)
        ax_strip = fig.add_subplot(gs[0, :])
        ax_beat = fig.add_subplot(gs[1, 0])
        ax_tally = fig.add_subplot(gs[1, 1])
        graph_scatter = None

    t_full = np.arange(len(strip)) / fs
    (line_strip,) = ax_strip.plot([], [], color="#1A1A1A", linewidth=1.1)
    _style_axis(ax_strip, fs, f"Record {record} - live classification")
    ax_strip.set_ylim(float(strip.min()) - 0.6, float(strip.max()) + 1.2)
    ax_strip.set_ylabel("amplitude (z)")

    (line_true,) = ax_beat.plot([], [], color="#1A1A1A", linewidth=1.4, label="target")
    (line_recon,) = ax_beat.plot([], [], color="#0072B2", linewidth=1.4, alpha=0.85, label="predicted")
    _style_axis(ax_beat, fs, "Regression head: current beat")
    ax_beat.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_beat.set_xlabel("samples")

    ax_tally.axis("off")
    markers: list = []
    status = ax_strip.text(
        0.005, 0.965, "", transform=ax_strip.transAxes, fontsize=10, va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#999999", alpha=0.92),
    )

    def frame(k: int):
        lo = k * step
        hi = min(lo + window_samples, len(strip))
        line_strip.set_data(t_full[lo:hi], strip[lo:hi])
        ax_strip.set_xlim(t_full[lo], t_full[lo] + seconds_visible)

        for m in markers:
            m.remove()
        markers.clear()

        seen = [b for b in beats if b.position <= hi]
        for b in seen:
            if not (lo <= b.position <= hi):
                continue
            color = OK_COLOR if b.correct else BAD_COLOR
            markers.append(
                ax_strip.axvline(b.position / fs, color=color, alpha=0.35, linewidth=1.2)
            )
            markers.append(
                ax_strip.text(
                    b.position / fs, float(strip.max()) + 0.55, CLASS_NAMES[b.pred_class],
                    color=color, fontsize=9, fontweight="bold", ha="center",
                )
            )

        if seen:
            cur = seen[-1]
            line_true.set_data(np.arange(len(cur.target)), cur.target)
            line_recon.set_data(np.arange(len(cur.recon)), cur.recon)
            ax_beat.set_xlim(0, len(cur.target))
            span = np.concatenate([cur.target, cur.recon])
            ax_beat.set_ylim(float(span.min()) - 0.3, float(span.max()) + 0.3)

            if graph_scatter is not None and cur.vm is not None:
                activation = _graph_step_for(cur, hi, fs, len(cur.window))
                graph_scatter.set_array(activation)

            correct = sum(b.correct for b in seen)
            status.set_text(
                f"beat {len(seen):3d}/{len(beats)}   pred {CLASS_NAMES[cur.pred_class]} "
                f"({cur.confidence:4.0%})   true {CLASS_NAMES[cur.true_class]}\n"
                f"running accuracy {correct / len(seen):5.1%}"
            )
            _draw_tally(ax_tally, seen)
        artists = [line_strip, line_true, line_recon, status, *markers]
        if graph_scatter is not None:
            artists.append(graph_scatter)
        return artists

    anim = FuncAnimation(fig, frame, frames=n_frames, blit=False, interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=n_frames, desc="rendering gif", unit="frame", leave=False) as bar:
        anim.save(
            str(out_path), writer=PillowWriter(fps=fps),
            progress_callback=lambda i, n: bar.update(1),
        )
    plt.close(fig)
    return out_path


def _draw_tally(ax, seen: list[BeatPrediction]) -> None:
    ax.clear()
    ax.axis("off")
    counts: dict[int, list[int]] = {}
    for b in seen:
        hit, tot = counts.get(b.true_class, [0, 0])
        counts[b.true_class] = [hit + int(b.correct), tot + 1]

    ax.text(0, 1.0, "per-class tally", fontsize=10, fontweight="bold", va="top")
    for row, (cls, (hit, tot)) in enumerate(sorted(counts.items())):
        y = 0.82 - row * 0.16
        name = CLASS_NAMES[cls]
        ax.text(0.0, y, name, fontsize=10, color=CLASS_COLORS[name], fontweight="bold", va="center")
        ax.add_patch(plt.Rectangle((0.16, y - 0.045), 0.62, 0.09, color="#E8E8E8"))
        if tot:
            ax.add_patch(
                plt.Rectangle((0.16, y - 0.045), 0.62 * hit / tot, 0.09, color=CLASS_COLORS[name])
            )
        ax.text(0.82, y, f"{hit}/{tot}", fontsize=9, va="center", fontfamily="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)


def render_summary(
    strip: np.ndarray, beats: list[BeatPrediction], fs: int, out_path: Path, record: str
) -> Path:
    """Static end-state figure: full strip, reconstruction overlay, and metrics."""
    fig = plt.figure(figsize=(14, 8), dpi=130)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.25)

    ax_strip = fig.add_subplot(gs[0, :])
    t = np.arange(len(strip)) / fs
    ax_strip.plot(t, strip, color="#1A1A1A", linewidth=0.9)
    _style_axis(ax_strip, fs, f"Record {record} - all {len(beats)} beats, model vs cardiologist")
    top = float(strip.max())
    for b in beats:
        color = OK_COLOR if b.correct else BAD_COLOR
        ax_strip.axvline(b.position / fs, color=color, alpha=0.3, linewidth=1.0)
        ax_strip.text(
            b.position / fs, top + 0.35, CLASS_NAMES[b.pred_class],
            color=color, fontsize=8, fontweight="bold", ha="center",
        )
        if not b.correct:
            ax_strip.text(
                b.position / fs, top + 0.95, f"true {CLASS_NAMES[b.true_class]}",
                color="#555555", fontsize=6.5, ha="center",
            )
    ax_strip.set_ylim(float(strip.min()) - 0.5, top + 1.4)
    ax_strip.set_xlabel("time (s)")

    # Three example reconstructions: best, median and worst by correlation.
    scored = [b for b in beats if b.has_target]
    if scored:
        corrs = [float(np.corrcoef(b.target, b.recon)[0, 1]) for b in scored]
        order = np.argsort(corrs)
        picks = [("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])]
        for col, (label, k) in enumerate(picks):
            ax = fig.add_subplot(gs[1, col])
            b = scored[k]
            ax.plot(b.target, color="#1A1A1A", linewidth=1.3, label="target")
            ax.plot(b.recon, color="#0072B2", linewidth=1.3, alpha=0.85, label="predicted")
            _style_axis(ax, fs, f"{label} reconstruction (r={corrs[k]:.3f})")
            if col == 0:
                ax.legend(fontsize=8, loc="upper right")

    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")
    correct = sum(b.correct for b in beats)
    lines = [
        f"beats shown: {len(beats)}    correct: {correct}    accuracy: {correct / len(beats):.1%}",
    ]
    if scored:
        sse = sum(float(((b.recon - b.target) ** 2).sum()) for b in scored)
        sst = sum(float((b.target**2).sum()) for b in scored)
        mean_r = float(np.mean(corrs))
        lines.append(
            f"regression on {len(scored)} beats:   PRD {100 * np.sqrt(sse / (sst + 1e-8)):.2f}%"
            f"    mean Pearson {mean_r:.4f}"
        )
    wrong = [b for b in beats if not b.correct]
    if wrong:
        pairs = sorted({f"{CLASS_NAMES[b.true_class]}->{CLASS_NAMES[b.pred_class]}" for b in wrong})
        lines.append(f"misclassifications: {', '.join(pairs)}")
    lines.append(
        "This is a single record, so it illustrates behaviour -- it is not a metric. "
        "Use the test-set numbers for that."
    )
    ax_stats.text(
        0.0, 0.9, "\n".join(lines), fontsize=10, va="top", fontfamily="monospace", linespacing=1.7
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def visualize_record(
    model,
    dataset: MITBIHBeatDataset,
    device,
    record: str | None = None,
    n_beats: int = 25,
    out_dir: str | Path = "figures",
    fps: int = 20,
    make_gif: bool = True,
    focus_class: int | None = None,
) -> dict[str, Path]:
    """Full pipeline: pick a record, predict, render the GIF and the summary.

    If `focus_class` is set and no record is given, the record holding the most
    beats of that class is chosen automatically.
    """
    if record is None:
        if focus_class is not None:
            record = _record_with_class(dataset, focus_class) or dataset.sample_records[0]
        else:
            record = dataset.sample_records[0]
    out_dir = Path(out_dir)
    strip, beats, fs = collect_predictions(
        model, dataset, device, record, n_beats, focus_class=focus_class
    )

    outputs = {
        "summary": render_summary(strip, beats, fs, out_dir / f"{record}_summary.png", record)
    }
    if make_gif:
        outputs["realtime"] = render_realtime(
            strip, beats, fs, out_dir / f"{record}_realtime.gif", record, fps=fps,
            graph_spec=getattr(model, "graph", None),
        )
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs

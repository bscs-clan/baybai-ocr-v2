"""Separate-process live training plot window and final plot export."""
import argparse
import json
import time
from pathlib import Path

_BAYBAYIN_MARK_LABELS = {
    "᜔": "virama (+)",  # U+1714
    "ᜓ": "bottom (•)",  # U+1713
    "ᜒ": "top (•)",     # U+1712
}


def _configure_fonts(matplotlib):
    """Load the bundled Philippine script font and return its glyph coverage."""
    from matplotlib import font_manager

    tagalog_font = Path(__file__).resolve().parents[2] / "fonts" / "Noto_Sans_Tagalog" / "NotoSansTagalog-Regular.ttf"
    if tagalog_font.is_file():
        font_manager.fontManager.addfont(str(tagalog_font))
    else:
        try:
            tagalog_font = Path(font_manager.findfont("Noto Sans Tagalog", fallback_to_default=False))
        except ValueError:
            tagalog_font = None
    # Use a full Latin font for graph text and numeric axes. Apply Tagalog only
    # to Baybayin tick labels below, since the script font has limited Latin coverage.
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    if tagalog_font:
        tagalog_coverage = set(font_manager.get_font(str(tagalog_font)).get_charmap())
    else:
        tagalog_coverage = set()
    fallback_font = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    fallback_coverage = set(font_manager.get_font(fallback_font).get_charmap())
    return tagalog_coverage | fallback_coverage


def _matrix_tick_labels(labels, glyph_coverage):
    display_labels = []
    for label in labels:
        if label == " ":
            display_labels.append("space")
        elif label == "<inserted>":
            display_labels.append("inserted")
        elif label == "<deleted>":
            display_labels.append("deleted")
        elif label in _BAYBAYIN_MARK_LABELS:
            display_labels.append(_BAYBAYIN_MARK_LABELS[label])
        elif ord(label) in glyph_coverage:
            display_labels.append(label)
        else:
            display_labels.append(f"U+{ord(label):04X}")
    return display_labels


def _set_matrix_ticks(axis, labels, glyph_coverage):
    from matplotlib import font_manager

    display_labels = _matrix_tick_labels(labels, glyph_coverage)
    axis.set_xticks(range(len(labels)), labels=display_labels, rotation=90, fontsize=8)
    axis.set_yticks(range(len(labels)), labels=display_labels, fontsize=8)

    font_path = Path(__file__).resolve().parents[2] / "fonts" / "Noto_Sans_Tagalog" / "NotoSansTagalog-Regular.ttf"
    if font_path.is_file():
        baybayin_font = font_manager.FontProperties(fname=str(font_path))
    else:
        baybayin_font = font_manager.FontProperties(family="Noto Sans Tagalog")
    for raw_label, tick in zip(labels, axis.get_xticklabels()):
        if (raw_label not in _BAYBAYIN_MARK_LABELS and len(raw_label) == 1
                and 0x1700 <= ord(raw_label) <= 0x1736 and ord(raw_label) in glyph_coverage):
            tick.set_fontproperties(baybayin_font)
    for raw_label, tick in zip(labels, axis.get_yticklabels()):
        if (raw_label not in _BAYBAYIN_MARK_LABELS and len(raw_label) == 1
                and 0x1700 <= ord(raw_label) <= 0x1736 and ord(raw_label) in glyph_coverage):
            tick.set_fontproperties(baybayin_font)


def _read_records(path: Path):
    records = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records


def _build_plots(records, output_dir: Path, interactive: bool):
    import matplotlib
    import matplotlib.pyplot as plt

    glyph_coverage = _configure_fonts(matplotlib)
    batches = [r for r in records if r.get("kind") == "batch" and r.get("train_loss") is not None]
    epochs = [r for r in records if r.get("kind") == "epoch"]
    train_x = [r.get("global_step", i + 1) for i, r in enumerate(batches)]
    train_y = [r["train_loss"] for r in batches]
    val_loss = [(r["epoch"], r["val_loss"]) for r in epochs if r.get("val_loss") is not None]
    quality = [(r["epoch"], r.get("val_cer"), r.get("word_accuracy")) for r in epochs
               if r.get("val_cer") is not None or r.get("word_accuracy") is not None]
    matrix_record = next((r for r in reversed(epochs) if r.get("confusion_matrix")), None)

    fig_loss, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(train_x, train_y, label="Training loss (batch average)", linewidth=1.2)
    ax.set(title="Training loss", xlabel="Optimizer step", ylabel="CTC loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig_val_loss, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if val_loss:
        ax.plot([x for x, _ in val_loss], [y for _, y in val_loss], marker="o", label="Validation loss")
    ax.set(title="Validation loss", xlabel="Epoch", ylabel="CTC loss")
    ax.grid(True, alpha=0.25)
    if val_loss:
        ax.legend(loc="best")

    fig_quality, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    cer = [(x, y) for x, y, _ in quality if y is not None]
    accuracy = [(x, y) for x, _, y in quality if y is not None]
    if cer:
        ax.plot([x for x, _ in cer], [y for _, y in cer], marker="o", label="Validation CER (lower is better)")
    if accuracy:
        ax.plot([x for x, _ in accuracy], [y for _, y in accuracy], marker="o", label="Exact line accuracy")
    ax.set(title="Validation quality", xlabel="Epoch", ylabel="Score")
    ax.grid(True, alpha=0.25)
    if cer or accuracy:
        ax.legend(loc="best")
    ax.set_ylim(-0.03, 1.03)

    fig_matrix, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    if matrix_record:
        matrix = matrix_record["confusion_matrix"]
        labels = matrix_record["confusion_labels"]
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)
        _set_matrix_ticks(ax, labels, glyph_coverage)
        ax.set(title=f"Validation character confusion matrix · epoch {matrix_record['epoch']}",
               xlabel="Predicted character", ylabel="Reference character")
        fig_matrix.colorbar(image, ax=ax, label="Character count")
    else:
        ax.set_title("Validation character confusion matrix")
        ax.text(0.5, 0.5, "Waiting for validation results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    figures = [(fig_loss, "training_loss.png"), (fig_val_loss, "validation_loss.png"),
               (fig_quality, "validation_quality.png"), (fig_matrix, "confusion_matrix.png")]
    if interactive:
        return figures
    output_dir.mkdir(parents=True, exist_ok=True)
    for fig, filename in figures:
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
    return figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--done", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = Path(args.metrics)
    done = Path(args.done)
    output_dir = Path(args.output_dir)

    import matplotlib
    glyph_coverage = _configure_fonts(matplotlib)
    # Try an interactive backend first. If the machine has no desktop display,
    # use a non-interactive backend and still save the final graphs.
    try:
        import matplotlib.pyplot as plt
        # Build the helper figures without Matplotlib opening each one as a
        # separate window. Only the combined dashboard below should be shown.
        plt.ioff()
        if str(matplotlib.get_backend()).lower() in {"agg", "pdf", "ps", "svg", "template"}:
            raise RuntimeError(f"Matplotlib selected non-interactive backend {matplotlib.get_backend()}")
        figures = _build_plots([], output_dir, interactive=True)
    except Exception as exc:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        print(f"Live plot window unavailable ({exc}); final graphs will still be saved.", flush=True)
        while not done.exists():
            time.sleep(0.5)
        _build_plots(_read_records(metrics), output_dir, interactive=False)
        return

    fig_loss, fig_val_loss, fig_quality, fig_matrix = [fig for fig, _ in figures]
    # Keep a single resizable window with all live graphs and the confusion matrix.
    for fig in (fig_val_loss, fig_quality, fig_matrix):
        plt.close(fig)
    fig = fig_loss
    fig.clear()
    fig.set_constrained_layout(False)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.15), hspace=0.28, wspace=0.28,
                            left=0.07, right=0.97, top=0.96, bottom=0.08)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[1, 0])
    ax_quality = fig.add_subplot(grid[2, 0])
    ax_matrix = fig.add_subplot(grid[:, 1])
    fig.set_size_inches(15, 12, forward=True)
    fig.canvas.manager.set_window_title("Live training graphs")
    lines = [ax_loss.plot([], [], label="Training batch loss")[0],
             ax_val.plot([], [], marker="o", label="Validation loss")[0],
             ax_quality.plot([], [], marker="o", label="Validation CER (lower is better)")[0],
             ax_quality.plot([], [], marker="o", label="Exact line accuracy")[0]]
    ax_loss.set(title="Training loss", xlabel="Optimizer step", ylabel="CTC loss")
    ax_val.set(title="Validation loss", xlabel="Epoch", ylabel="CTC loss")
    ax_quality.set(title="Validation quality", xlabel="Epoch", ylabel="Score", ylim=(-0.03, 1.03))
    ax_matrix.set_title("Validation character confusion matrix")
    ax_matrix.text(0.5, 0.5, "Waiting for validation results", ha="center", va="center",
                   transform=ax_matrix.transAxes)
    ax_matrix.set_axis_off()
    for ax in (ax_loss, ax_val, ax_quality):
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    state = {"signature": None, "finished": False, "matrix_colorbar": None}

    def refresh():
        records = _read_records(metrics)
        batches = [r for r in records if r.get("kind") == "batch" and r.get("train_loss") is not None]
        epochs = [r for r in records if r.get("kind") == "epoch"]
        signature = (len(batches), len(epochs))
        if signature != state["signature"]:
            state["signature"] = signature
            lines[0].set_data([r.get("global_step", i + 1) for i, r in enumerate(batches)], [r["train_loss"] for r in batches])
            lines[1].set_data([r["epoch"] for r in epochs if r.get("val_loss") is not None], [r["val_loss"] for r in epochs if r.get("val_loss") is not None])
            lines[2].set_data([r["epoch"] for r in epochs if r.get("val_cer") is not None], [r["val_cer"] for r in epochs if r.get("val_cer") is not None])
            lines[3].set_data([r["epoch"] for r in epochs if r.get("word_accuracy") is not None], [r["word_accuracy"] for r in epochs if r.get("word_accuracy") is not None])
            for ax in (ax_loss, ax_val):
                ax.relim()
                ax.autoscale_view()
            ax_quality.relim()
            ax_quality.autoscale_view(scalex=True, scaley=False)
            ax_quality.set_ylim(-0.03, 1.03)
            matrix_record = next((r for r in reversed(epochs) if r.get("confusion_matrix")), None)
            if matrix_record:
                ax_matrix.clear()
                matrix = matrix_record["confusion_matrix"]
                labels = matrix_record["confusion_labels"]
                matrix_image = ax_matrix.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)
                matrix_image.set_clim(0, max(1, max(max(row) for row in matrix)))
                _set_matrix_ticks(ax_matrix, labels, glyph_coverage)
                ax_matrix.set(title=f"Validation character confusion matrix · epoch {matrix_record['epoch']}",
                              xlabel="Predicted character", ylabel="Reference character")
                if state["matrix_colorbar"] is None:
                    state["matrix_colorbar"] = fig.colorbar(matrix_image, ax=ax_matrix, label="Character count")
                else:
                    state["matrix_colorbar"].update_normal(matrix_image)
            # This callback already runs on the GUI timer thread. Force the
            # redraw here so TkAgg does not defer/drop it while pyplot is in
            # non-interactive mode.
            fig.canvas.draw()
        if done.exists() and not state["finished"]:
            state["finished"] = True
            output_dir.mkdir(parents=True, exist_ok=True)
            # Export separate figures from the final metrics and leave the live
            # window open so the completed curves can still be inspected.
            _build_plots(records, output_dir, interactive=False)
        return True

    def safe_refresh():
        try:
            return refresh()
        except Exception as exc:
            # Keep the timer alive if a single read or redraw fails; surface the
            # error in the training console instead of silently freezing plots.
            print(f"Live graph refresh failed: {exc!r}", flush=True)
            return True

    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(safe_refresh)
    timer.start()
    refresh()
    # Keep this child process alive to service GUI events. Without block=True,
    # pyplot may return immediately and the window disappears with the process.
    plt.show(block=True)
    # Some GUI backends close their event loop before the done marker arrives.
    if not state["finished"]:
        _build_plots(_read_records(metrics), output_dir, interactive=False)


if __name__ == "__main__":
    main()

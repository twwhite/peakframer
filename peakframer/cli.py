"""CLI entrypoint for peakframer — parses arguments and orchestrates the pipeline."""

from pathlib import Path
from typing import Annotated

import cv2
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from peakframer import __version__
from peakframer.utils import detect_device, get_logger, set_log_level

logger = get_logger(__name__)

app = typer.Typer(
    name="peakframer",
    help="Extract maximally diverse frames from video using visual embeddings.",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"peakframer {__version__}")
        raise typer.Exit()


@app.command()
def run(
    video: Path = typer.Argument(
        ...,
        help="Path to input video file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    count: int = typer.Option(
        10, "--count", "-n", help="Number of diverse frames to extract."
    ),
    output: Path = typer.Option(
        Path("./peakframer_out"), "--output", "-o", help="Output directory."
    ),
    sample_rate: int | None = typer.Option(
        None,
        "--sample-rate",
        "-s",
        help="Decode every Nth frame. Auto-tuned if not set.",
    ),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU inference."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    from peakframer.embedder import CLIPEmbedder
    from peakframer.extractor import extract_frames, get_video_meta, suggest_sample_rate
    from peakframer.sampler import select_diverse_indices

    set_log_level(debug)

    """
        → get_video_meta()
        → extract_frames()
        → embedder.embed()
        → select_diverse_indices()
        → save frames
    """

    logger.debug(f"Video: {video}")
    logger.debug(f"Extracting {count} frames, sample rate: every {sample_rate} frames")
    logger.debug(f"Output: {output}")
    logger.debug(f"Device: {'cpu' if cpu else 'auto'}")

    device = detect_device(force_cpu=cpu)
    logger.debug(f"Detected device: {device}")
    output.mkdir(parents=True, exist_ok=True)

    meta = get_video_meta(video)
    logger.info(
        f"Video: {meta.path.name} — {meta.duration_seconds:.1f}s, "
        "{meta.width}x{meta.height} @ {meta.fps:.1f}fps"
    )

    if sample_rate is None:
        sample_rate = suggest_sample_rate(meta.total_frames, count)
        logger.info(f"Auto-tuned sample rate: every {sample_rate} frames")

    logger.info(
        f"Candidate frames: ~{meta.total_frames // sample_rate} "
        "(every {sample_rate} frames)"
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
        progress.add_task("Decoding frames...")
        frames = extract_frames(video, sample_rate=sample_rate)

    candidate_count = len(frames)
    if candidate_count < count * 5:
        logger.warning(
            f"Only {candidate_count} frames were sampled from the video, but you "
            f"requested {count} diverse frames.\nFor best results, lower --sample-rate "
            f"to give the tool more frames to choose from, or use a longer video."
            f"\n(ratio recommended: at least {count * 5} (--count × --sample-rate))."
            f"\n(Omit --sample-rate to use auto-sampling)."
        )
        input("Press enter to continue...")

    embedder = CLIPEmbedder(device=device)
    images = [f.image for f in frames]

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Embedding frames...", total=len(frames))
        embeddings = embedder.embed(images)
        progress.update(task, completed=len(frames))

    selected_indices = select_diverse_indices(embeddings, count=count)
    for rank, idx in enumerate(sorted(selected_indices, key=lambda i: frames[i].index)):
        frame = frames[idx]
        ts = frame.timestamp_ms / 1000.0
        filename = f"frame_{rank:04d}_t{ts:.3f}s.jpg"
        cv2.imwrite(str(output / filename), frame.image)

    logger.info(f"Saved {len(selected_indices)} frames to {output}")


if __name__ == "__main__":
    app()

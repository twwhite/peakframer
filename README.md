# peakframer

Extract maximally diverse frames from video using visual embeddings.

![peakframer banner](assets/banner.jpg)

## Install
```bash
uv sync --group dev
```

## Usage
```bash
peakframer video.mp4 --count 50
peakframer video.mp4 --count 100 --output ./frames --sample-rate 10
peakframer video.mp4 --count 50 --debug
```

## How it works

1. Decode every Nth frame from the video
2. Embed each frame with CLIP (ViT-B/32)
3. Cluster embeddings with k-means (k = count)
4. Save the frame closest to each centroid

## License

Apache 2.0

## Disclaimer

This project was developed with the assistance of AI tools.

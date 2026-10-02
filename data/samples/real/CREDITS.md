# Credits

## hummingbird_feeder.mp4 (not in the repository)

A hummingbird flies in to a red nectar feeder, perches on it for about 8 s and flies off, filmed from a static camera. The clip runs `docs/demo-real.gif` and the real-clip test. It is not committed: `python scripts/fetch_real_clip.py` downloads it to this directory (gitignored).

| | |
|---|---|
| Source page | [Hummingbird landing on a feeder](https://pixabay.com/videos/hummingbird-bird-feeder-landing-110462/) on Pixabay, video 110462, published 2022-03-17, 3840×2160, 30 fps, 16.6 s |
| File downloaded | `https://cdn.pixabay.com/video/2022/03/12/110462-689510229_small.mp4`, the 1920×1080 rendition, 10,731,827 bytes, SHA-256 `12096e93a385436ae9a07ed185267b98f212f390653667234d3c969a65008651` |
| Author | ZacharyCrespin (Pixabay user) |
| License | **Pixabay Content License** ([summary](https://pixabay.com/service/license-summary/), [full terms](https://pixabay.com/service/terms/)). Free to use and to modify, commercially or not, with no attribution required and no non-commercial or no-derivatives clause. It forbids selling or distributing the content on a "standalone" basis, unchanged; that is why the clip itself is downloaded rather than committed. Read on the file page on 2026-10-02: "Free for use under the Pixabay Content License". |
| Trim | 0 s to 16.6 s, the whole clip: empty feeder for 3 s, the visit, empty again from about 14 s. Video stream copied with ffmpeg (no re-encode); the audio track is dropped. |

**Changes:** none to the picture. `docs/demo-real.gif` is a derived work: frames of this clip next to the pipeline's annotated output, scaled down, with captions and a sped-up middle section.

Attribution is not legally required under the Pixabay Content License. It is given here so the source can be checked.

## roi.json

The region of interest for this clip (x=248 y=164 w=1078 h=916 on 1920×1080), for `main.py --roi-file`. It was computed, not drawn: `python scripts/fetch_real_clip.py --roi` runs YOLOv8n on every 5th frame and takes the union of the boxes where the bird perches, padded by 10%. See `docs/observations.md`, "Real-clip findings".

# Credits

## house_sparrow_female.png

A cutout of a photograph of a female house sparrow (*Passer domesticus*) perched on a railing. `scripts/make_synth_video.py` composites it into the synthetic quickstart clip.

| | |
|---|---|
| Source file | [File:House sparrow (female) (50108651573).jpg](https://commons.wikimedia.org/wiki/File:House_sparrow_(female)_(50108651573).jpg) on Wikimedia Commons, 3000×2003, SHA-1 `559af14442251a6edbba59f3e7b2405a00de0b57` |
| Original | [flickr.com/photos/usfwsmidwest/50108651573](https://www.flickr.com/photos/usfwsmidwest/50108651573), U.S. Fish and Wildlife Service – Midwest Region, taken 2020-07-10 |
| Author | Courtney Celley / USFWS (per the file description: "Photo by Courtney Celley/USFWS.") |
| License | **Public domain.** The work of a U.S. Fish and Wildlife Service employee, made as part of that person's official duties (Commons template `PD-USGov-FWS`). The Commons license review (FlickreviewR 2, 2025-03-24) confirmed that the Flickr original is marked with the Public Domain Mark. Both were read on the Commons file page on 2026-09-30. |

**Changes:** cropped to the bird and cut out of its background with OpenCV GrabCut, with the edge feathered, then scaled to 320 px wide and saved as RGBA PNG. `python scripts/make_bird_cutout.py <source jpg>` reproduces it. A thin sliver of the railing remains under the tail.

Attribution is not legally required for a public-domain work. It is given here so the source can be checked.

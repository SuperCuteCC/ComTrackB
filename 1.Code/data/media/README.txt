Videos are not bundled, for the same reason the 10.35 GB base weight is not bundled:
they are public files and are pinned by hash instead of copied.

Place the 568 MP4 files here as <video_id>.mp4 (for example simulation_0007.mp4), or
keep them anywhere and pass the directory to verification/prepare_runtime.py as
--videos-dir. Every file is checked against data/VIDEO_MANIFEST_SHA256.csv before use.

Source: PaulineLi/QuantiPhy, revision 072ede13a186ee71d76ae628692cbdb11c4e1721

  huggingface-cli download PaulineLi/QuantiPhy --repo-type dataset \
    --revision 072ede13a186ee71d76ae628692cbdb11c4e1721 --local-dir /data/quantiphy

The 568 files used by this submission total 529,911,876 bytes. data/videos/ keeps the
dataset card, the original download record and the face-blur logs.

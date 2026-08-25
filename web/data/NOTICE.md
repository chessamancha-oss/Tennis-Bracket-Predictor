# Professional player data notice

Official ranking positions and points are a versioned snapshot from the ATP and WTA ranking pages on 2026-08-18.

Historical match aggregates are derived from the public tennis datasets compiled by Jeff Sackmann and preserved by the `tennis-sackmann-archive`. Those datasets are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This project uses them for non-commercial research, attributes the original compiler, identifies the derived fields, and carries the same terms for the derived data.

- Original compiler: Jeff Sackmann / Tennis Abstract
- Original ATP source: https://github.com/JeffSackmann/tennis_atp
- Original WTA source: https://github.com/JeffSackmann/tennis_wta
- Archival mirror: https://github.com/Aneeshers/tennis-sackmann-archive
- History cutoff in this build: 2026-05-25
- Ranking snapshot in this build: 2026-08-18
- Derived historical catalogue: 7,255 ATP/WTA profiles spanning 1967–2026

The live tournament interface separately reads short-lived scoreboard data from ESPN. That response is not redistributed in the committed player dataset and is refreshed on demand. The interface links to official tournament sources for verification.

The context interface reads U.S. weather observations from the public National Weather Service API and global venue geocoding, forecasts, and fallback weather from Open-Meteo under its published attribution terms. It uses Google News RSS as a discovery layer for this private, personal, non-commercial research deployment and links each surfaced headline to its publisher. Headlines and weather responses are not committed to or redistributed with the player dataset. A separate rights review is required before changing this deployment to public or commercial use.

The model and interface are not affiliated with or endorsed by the ATP, WTA, ESPN, players, tournaments, or data contributors.

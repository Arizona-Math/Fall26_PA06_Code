# Archived utility data

`tucsonAZ.csv` and `phoenixAZ.csv` are unchanged copies of the files distributed
with the earlier Math 589A M3 seasonal difference-equation assignment.
They are bundled for reproducible offline coursework.

Original source: Renee Obringer, *Climate Analogs - Water-Energy Nexus Analysis*,
version 1.0.0 (2023), [Zenodo, DOI 10.5281/zenodo.7826348](https://doi.org/10.5281/zenodo.7826348).
The [upstream data directory](https://github.com/reneeobringer/ClimateAnalogs_WEN/tree/v1.0.0/UtilityData)
and its [units description](https://github.com/reneeobringer/ClimateAnalogs_WEN/blob/v1.0.0/UtilityData/README)
document the processed monthly observations.

| Column | Meaning |
|---|---|
| `year`, `month` | Calendar month |
| `wateruse` | Monthly water use, gallons per capita |
| `elecuse` | Monthly electricity use, MWh per capita |

PA06 uses `elecuse`. Tucson has 144 consecutive months, January 2007 through
December 2018. Phoenix has 138, January 2007 through June 2018. Do not align
their final rows as though they were the same calendar month. No values have
been interpolated, rescaled, or replaced for this assignment.

These are historical observations, not forecasts or current utility figures.
The supplied source attribution is retained independently of course-code
authorship. Refer to the upstream release for its terms; no new data license
is asserted by this assignment package.

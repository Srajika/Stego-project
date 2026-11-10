# Phase 5B detection study summary

| Study item | Value |
|---|---:|
| Usable covers | 23 |
| DCT embedding | quality 75; coefficient position (3, 2); clamp Y=[6, 249] |
| Copy-corrected DCT p-value | chi2.sf(statistic / 3, 1); counts the three repeated copies once |
| Default fire threshold | p >= 0.01; sequential default uses the detector verdict |
| Full-run duration | 1,632.542 s (27 min 12.5 s) |

## 1. Detection rate k/n: uniform random bytes

| Method / detector | 0% | 5% | 10% | 25% | 50% | 90% | 95% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain LSB detect | 0/23 | 0/23 | 0/23 | 0/23 | 0/23 | 3/23 | 3/23 | 22/23 |
| Plain LSB sequential (20) | 0/23 | 12/23 | 22/23 | 22/23 | 22/23 | 21/23 | 22/23 | 22/23 |
| Plain LSB sequential (100) | 1/23 | 21/23 | 21/23 | 20/23 | 20/23 | 20/23 | 21/23 | 21/23 |
| Password LSB detect | 0/23 | - | 0/23 | - | 0/23 | 1/23 | 5/23 | 23/23 |
| DCT parity original p | 1/23 | - | 0/14 | - | 1/23 | 2/23 | 1/23 | 14/23 |
| DCT parity copy-corrected p | 2/23 | - | 0/14 | - | 2/23 | 2/23 | 8/23 | 22/23 |

| Rate note | DCT 0% uses the processed control; raw-cover false alarms are in section 2. DCT uniform 10% is 0/14 among feasible covers; nine covers could not fit the 69-byte framing overhead. Unrun rates are n/a. |

## Printable text at 100%

| Method / detector | Detection k/n |
|---|---:|
| Plain LSB detect | 0/23 |
| Plain LSB sequential (20) | 0/23 |
| Plain LSB sequential (100) | 1/23 |
| Password LSB detect | 0/23 |
| DCT parity original p | 0/23 |
| DCT parity copy-corrected p | 2/23 |

## 2. False alarms on clean raw covers and valid raw crops

| Detector | Full covers | Crops 256x192 | Crops 512x384 |
|---|---:|---:|---:|
| Plain LSB detect | 0/23 (0.0%) | 1/115 (0.9%) | 0/105 (0.0%) |
| Plain LSB sequential (20) | 0/23 (0.0%) | 15/115 (13.0%) | 6/105 (5.7%) |
| Plain LSB sequential (100) | 1/23 (4.3%) | 29/115 (25.2%) | 13/105 (12.4%) |
| Password LSB detect | 0/23 (0.0%) | 1/115 (0.9%) | 0/105 (0.0%) |
| DCT parity original p | 1/23 (4.3%) | 21/115 (18.3%) | 13/105 (12.4%) |
| DCT parity copy-corrected p | 2/23 (8.7%) | 25/115 (21.7%) | 13/105 (12.4%) |

| Crop coverage | 256x192: 115 valid crops; 512x384: 105 valid crops. Two covers were too small for the larger crop. |

## 3. Smallest tested actual rate that fires

| Method / detector | Payload | Median minimum | Range | Never fires |
|---|---|---:|---:|---:|
| Plain LSB detect | Uniform bytes | 100.00% | 90.00-100.00% | 1/23 |
| Plain LSB detect | Printable text | - | - | 23/23 |
| Plain LSB sequential (20) | Uniform bytes | 5.00% | 5.00-10.00% | 1/23 |
| Plain LSB sequential (20) | Printable text | - | - | 23/23 |
| Plain LSB sequential (100) | Uniform bytes | 5.00% | 5.00-5.00% | 2/23 |
| Plain LSB sequential (100) | Printable text | 100.00% | 100.00-100.00% | 22/23 |
| Password LSB detect | Uniform bytes | 100.00% | 90.00-100.00% | 0/23 |
| Password LSB detect | Printable text | - | - | 23/23 |
| DCT parity original p | Uniform bytes | 99.98% | 50.00-100.00% | 6/23 |
| DCT parity original p | Printable text | - | - | 23/23 |
| DCT parity copy-corrected p | Uniform bytes | 99.89% | 50.00-100.00% | 1/23 |
| DCT parity copy-corrected p | Printable text | 99.93% | 99.86-100.00% | 21/23 |

## 4. ROC-style sweep at 100%: detection versus full-cover false alarms

| Method / detector | Payload | 0.001 (detect; FA) | 0.01 (detect; FA) | 0.05 (detect; FA) | 0.1 (detect; FA) |
|---|---|---:|---:|---:|---:|
| Plain LSB detect | Uniform bytes | 23/23 (100.0%); 0/23 (0.0%) | 22/23 (95.7%); 0/23 (0.0%) | 22/23 (95.7%); 0/23 (0.0%) | 22/23 (95.7%); 0/23 (0.0%) |
| Plain LSB detect | Printable text | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) |
| Plain LSB sequential (20) | Uniform bytes | 22/23 (95.7%); 0/23 (0.0%) | 22/23 (95.7%); 0/23 (0.0%) | 21/23 (91.3%); 0/23 (0.0%) | 20/23 (87.0%); 0/23 (0.0%) |
| Plain LSB sequential (20) | Printable text | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) |
| Plain LSB sequential (100) | Uniform bytes | 21/23 (91.3%); 1/23 (4.3%) | 21/23 (91.3%); 1/23 (4.3%) | 21/23 (91.3%); 1/23 (4.3%) | 21/23 (91.3%); 1/23 (4.3%) |
| Plain LSB sequential (100) | Printable text | 2/23 (8.7%); 1/23 (4.3%) | 1/23 (4.3%); 1/23 (4.3%) | 0/23 (0.0%); 1/23 (4.3%) | 0/23 (0.0%); 1/23 (4.3%) |
| Password LSB detect | Uniform bytes | 23/23 (100.0%); 0/23 (0.0%) | 23/23 (100.0%); 0/23 (0.0%) | 22/23 (95.7%); 0/23 (0.0%) | 21/23 (91.3%); 0/23 (0.0%) |
| Password LSB detect | Printable text | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) | 0/23 (0.0%); 0/23 (0.0%) |
| DCT parity original p | Uniform bytes | 18/23 (78.3%); 1/23 (4.3%) | 14/23 (60.9%); 1/23 (4.3%) | 11/23 (47.8%); 1/23 (4.3%) | 10/23 (43.5%); 1/23 (4.3%) |
| DCT parity original p | Printable text | 1/23 (4.3%); 1/23 (4.3%) | 0/23 (0.0%); 1/23 (4.3%) | 0/23 (0.0%); 1/23 (4.3%) | 0/23 (0.0%); 1/23 (4.3%) |
| DCT parity copy-corrected p | Uniform bytes | 23/23 (100.0%); 2/23 (8.7%) | 22/23 (95.7%); 2/23 (8.7%) | 18/23 (78.3%); 1/23 (4.3%) | 17/23 (73.9%); 1/23 (4.3%) |
| DCT parity copy-corrected p | Printable text | 3/23 (13.0%); 2/23 (8.7%) | 2/23 (8.7%); 2/23 (8.7%) | 1/23 (4.3%); 1/23 (4.3%) | 0/23 (0.0%); 1/23 (4.3%) |

| ROC note | Threshold sweep uses the CSV p-values; for sequential detectors this is the stored first-curve p-value. DCT corrected uses its corrected p-value. |

## 5. Crop false alarms versus content descriptors

| Detector | Flagged n | Flagged median Y SD | Flagged flat-block fraction | Flagged distinct RGB | Unflagged n | Unflagged median Y SD | Unflagged flat-block fraction | Unflagged distinct RGB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain LSB detect | 1 | 30.68 | 0.000 | 17752.0 | 219 | 44.92 | 0.072 | 19086.0 |
| Plain LSB sequential (20) | 21 | 46.00 | 0.000 | 34159.0 | 199 | 44.62 | 0.101 | 14908.0 |
| Plain LSB sequential (100) | 42 | 46.82 | 0.000 | 37695.0 | 178 | 43.86 | 0.137 | 10585.0 |
| Password LSB detect | 1 | 30.68 | 0.000 | 17752.0 | 219 | 44.92 | 0.072 | 19086.0 |
| DCT parity original p | 34 | 56.46 | 0.000 | 42957.0 | 186 | 39.89 | 0.121 | 11748.0 |
| DCT parity copy-corrected p | 38 | 57.80 | 0.000 | 43092.0 | 182 | 38.46 | 0.129 | 11326.0 |

| Full-cover false alarms | Image(s) |
|---|---|
| Plain LSB detect | - |
| Plain LSB sequential (20) | - |
| Plain LSB sequential (100) | bright_04.png |
| Password LSB detect | - |
| DCT parity original p | photos_02.png |
| DCT parity copy-corrected p | photos_02.png, photos_04.png |

## 6. Detection at 100% versus payload one-bit fraction

| Method / detector | Payload | n | Mean one-bit fraction (range) | Detection k/n |
|---|---|---:|---:|---:|
| Plain LSB detect | Uniform bytes | 23 | 0.499933 (0.499294-0.500711) | 22/23 |
| Plain LSB detect | Printable text | 23 | 0.474969 (0.474194-0.475384) | 0/23 |
| Plain LSB sequential (20) | Uniform bytes | 23 | 0.499933 (0.499294-0.500711) | 22/23 |
| Plain LSB sequential (20) | Printable text | 23 | 0.474969 (0.474194-0.475384) | 0/23 |
| Plain LSB sequential (100) | Uniform bytes | 23 | 0.499933 (0.499294-0.500711) | 21/23 |
| Plain LSB sequential (100) | Printable text | 23 | 0.474969 (0.474194-0.475384) | 1/23 |
| Password LSB detect | Uniform bytes | 23 | 0.500050 (0.499276-0.501350) | 23/23 |
| Password LSB detect | Printable text | 23 | 0.475011 (0.474523-0.475564) | 0/23 |
| DCT parity original p | Uniform bytes | 23 | 0.492044 (0.464286-0.503723) | 14/23 |
| DCT parity original p | Printable text | 23 | 0.467965 (0.439674-0.483852) | 0/23 |
| DCT parity copy-corrected p | Uniform bytes | 23 | 0.492044 (0.464286-0.503723) | 22/23 |
| DCT parity copy-corrected p | Printable text | 23 | 0.467965 (0.439674-0.483852) | 2/23 |

| Source | CSV: detection_results.csv; cross-checked against full_output.txt. The full report includes per-image timings and detailed diagnostics. |

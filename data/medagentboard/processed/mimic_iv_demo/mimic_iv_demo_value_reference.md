# MIMIC-IV Demo Value Reference

The formatted parquet stores categorical clinical channels as scalar-coded values.
Use this file to interpret the numeric values that appear in `mimic_iv_demo_formatted_ehr.parquet`.

## Capillary refill rate

| Stored value | Meaning / original labels |
| --- | --- |
| 0.0 | Normal capillary refill |
| 1.0 | Abnormal capillary refill |

## Glascow coma scale eye opening

| Stored value | Meaning / original labels |
| --- | --- |
| 1 | 1 No Response |
| 2 | 2 To pain, To Pain |
| 3 | 3 To speech, To Speech |
| 4 | 4 Spontaneously, Spontaneously |
| 0 | None |
| normal_value | 4 Spontaneously |

## Glascow coma scale motor response

| Stored value | Meaning / original labels |
| --- | --- |
| 1 | 1 No Response, No response |
| 2 | 2 Abnorm extensn, Abnormal extension |
| 3 | 3 Abnorm flexion, Abnormal Flexion |
| 4 | 4 Flex-withdraws, Flex-withdraws |
| 5 | 5 Localizes Pain, Localizes Pain |
| 6 | 6 Obeys Commands, Obeys Commands |
| normal_value | 6 Obeys Commands |

## Glascow coma scale total

| Stored value | Meaning / original labels |
| --- | --- |
| 10 | 10 |
| 11 | 11 |
| 12 | 12 |
| 13 | 13 |
| 14 | 14 |
| 15 | 15 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |
| 3-15 | Sum of eye opening, motor response, and verbal response scores |
| normal_value | 15 |

## Glascow coma scale verbal response

| Stored value | Meaning / original labels |
| --- | --- |
| 1 | 1 No Response, 1.0 ET/Trach, No Response, No Response-ETT |
| 2 | 2 Incomp sounds, Incomprehensible sounds |
| 3 | 3 Inapprop words, Inappropriate Words |
| 4 | 4 Confused, Confused |
| 5 | 5 Oriented, Oriented |
| normal_value | 5 Oriented |


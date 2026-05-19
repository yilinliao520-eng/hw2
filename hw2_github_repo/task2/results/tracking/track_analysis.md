# Tracking Analysis

- Total detections: 2857
- Unique tracking IDs: 78
- Video frames covered: 0 to 899
- Line crossings: 5

## Per-Class Observations

- car: 38 unique IDs
- van: 13 unique IDs
- three wheelers -CNG-: 11 unique IDs
- pickup: 10 unique IDs
- bus: 10 unique IDs
- minivan: 10 unique IDs
- motorbike: 10 unique IDs
- suv: 9 unique IDs
- human hauler: 4 unique IDs
- truck: 3 unique IDs
- bicycle: 1 unique IDs

## Longest Tracks

- ID 39 (car): frames 71-442, observed 372 frames, mean conf 0.842
- ID 143 (car): frames 396-667, observed 251 frames, mean conf 0.793
- ID 47 (car): frames 97-333, observed 237 frames, mean conf 0.849
- ID 118 (van): frames 316-579, observed 182 frames, mean conf 0.452
- ID 230 (car): frames 701-899, observed 173 frames, mean conf 0.617
- ID 253 (car): frames 758-899, observed 121 frames, mean conf 0.740
- ID 1 (car): frames 0-116, observed 115 frames, mean conf 0.829
- ID 199 (car): frames 635-760, observed 113 frames, mean conf 0.730
- ID 157 (car): frames 474-599, observed 102 frames, mean conf 0.652
- ID 83 (van): frames 221-318, observed 98 frames, mean conf 0.614

## Possible ID Switch Candidates

- No large trajectory jumps were detected by the simple heuristic.

## Notes for the Report

- Use the annotated video to manually inspect occlusion or crossing moments.
- If an object keeps the same Tracking ID before and after occlusion/crossing, record it as successful ID continuity.
- If two nearby objects exchange IDs or one object receives a new ID after reappearing, record it as an ID switch or ID loss case.
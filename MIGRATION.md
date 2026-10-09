# Migration and changed contracts

This is an engineering prototype, not a certified commercial release. Read REVIEW_FA.md and README_FA.md before deployment. legacy_docs/ contains the original documentation for comparison only.

## Renamed outputs

- possible_sleep -> resting_in_bed: apparent rest is not confirmed sleep or wellness.
- unsteady_gait -> possible_postural_sway: observed sway is not a validated clinical gait disorder or fall-risk score.
- possible_collapse -> rapid_descent_high_speed: fast movement does not diagnose syncope.
- New resting_in_chair and activity=moving labels distinguish recliner rest and non-walking movement.

CSV columns remain compatible, but enum values and event semantics changed. Update consumer allowlists. New run metadata has schema_version=2. The old sleep_still_s setting is retained for configuration compatibility; it controls only the bed-rest label, not a care deadline or sleep diagnosis.

## Geometry and policy

The new optional chair_polygon defaults to empty. Invalid numeric booleans, malformed/zero-area/self-intersecting polygons are rejected. Region membership now requires hip, shoulder and torso center together. Partial or multiple-region membership is ambiguous, not automatically safe bed membership. Define regions for the actual camera.

possible_fall now requires calibrated floor evidence. Without it, fast descent can produce descent_unconfirmed and a care review rather than claim floor contact. Inactivity is a new opt-in feature in care_config.json, not a feature present in the original code. A care_plan_id is required for inactivity or expected presence. The example config is NOT an approved individual plan.

## Durable incidents

An SQLite incident/outbox file is created beside the CSV unless --incident-db is supplied. upright_after_fall and floor_lying_ended only end visual episodes: neither closes the durable care incident. Track expiry, camera loss and restart also cannot close it. Operator ACK is not resolution. The local CLI relies on OS file permissions, not organizational identity authentication.

Run/track identity is scoped to each run; the code does not guess which resident a new track represents. Duplicate real-world incidents after tracking/restart remain possible and need human review. Use a separate database for historical replay; do not send replay events to the live response service.

## Delivery and operations

Run delivery.py as a separately supervised process for HTTPS dispatch and deadline handling. The optional --watch-site value must match live monitor --site-id. Without protected webhook settings, no external notification occurs. Receivers must authenticate the token, deduplicate Idempotency-Key and process ordered historical notifications by their time; a delayed opening event is not necessarily the current status. HTTP 2xx is only transport receipt, never human ACK. Delivery is at-least-once, not exactly-once.

Live monitoring now checks frame darkness, exact repetition, geometry changes and local inference age. These do not prove camera exposure freshness, detect every replay or compensate all camera motion. Geometry changes latch a recalibration requirement until a supervised restart. Reconnect inside the vision process is not implemented; a service manager is required.

CSV rotates to five backups by default. SQLite/audit and run metadata require a separately approved backup/retention/archiving plan. --source-env avoids camera secrets in argv, but environment access and third-party driver logs still need protection. Wall-clock changes can affect operational deadlines and must be controlled/tested in infrastructure.

## Test migration

All 48 original tests passed unchanged before editing. No original test method was removed. Expectations were intentionally updated for the three renamed labels, explicit floor calibration and ambiguous overlap contract. Another 62 tests cover the new policy, persistence, delivery, geometry, logging and integration. The final 110 tests pass in the documented environment; they do not establish real model accuracy.

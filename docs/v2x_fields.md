# Our warning fields next to ETSI DENM and SAE J2735 (P3-08)

**This is a field study, not a compliance claim.** Our `WarningMessage` and
`AssessedHazard` are Python objects carried by a simulated delivery model.
Nothing is ASN.1-encoded, signed or sent over a real radio. The tables below
show where each piece of information *would* go in the two standard hazard
messages, which of our fields have no standard home, and which standard
fields we do not produce. Use it to discuss how the framework could feed a
real V2X stack, not to claim that it does.

## Sources and how far they were checked

| Standard | What we checked against | Status |
|---|---|---|
| ETSI EN 302 637-3 v1.3.1 (DENM) | ETSI's published ASN.1 module `DENM-PDU-Descriptions` ([forge.etsi.org/rep/ITS/asn1/denm_en302637_3](https://forge.etsi.org/rep/ITS/asn1/denm_en302637_3), tag v1.3.1) | Field names, types and the `validityDuration` default checked against the ASN.1 |
| ETSI TS 102 894-2 v1.3.1 (Common Data Dictionary) | ETSI's published ASN.1 module `ITS-Container` ([forge.etsi.org/rep/ITS/asn1/cdd_ts102894_2](https://forge.etsi.org/rep/ITS/asn1/cdd_ts102894_2), tag v1.3.1) | Cause codes, sub-cause codes, `InformationQuality`, `RelevanceTrafficDirection`, `RelevanceDistance` checked against the ASN.1 |
| SAE J2735 (RoadSideAlert) | The SAE text is paywalled. The field list comes from the open-source `kitt` J2735 implementation ([kitt.hexdocs.pm, Kitt.Message.RSA](https://kitt.hexdocs.pm/Kitt.Message.RSA.html)) | **To confirm against the SAE document before citing.** Field names only; types and value ranges are not stated here |
| SAE J2540/2 (ITIS phrase lists) | Not accessed (paywalled) | No ITIS code numbers are given in this note |

The standards' prose descriptions of each data element were
not read in full; the mapping relies on the type definitions and their names.

## 1. Hazard classes to DENM cause codes

From `CauseCodeType` and the sub-cause types in ITS-Container v1.3.1.

| Our `HazardType` | DENM `eventType.causeCode` | Sub-cause (if one fits) | Fit |
|---|---|---|---|
| `accident` | `accident` (2) | `AccidentSubCauseCode` `unavailable` (0); `multiVehicleAccident` (1), `accidentInvolvingLorry` (3), `accidentInvolvingBus` (4) exist but our classifier does not extract them | Direct |
| `road_obstruction` | `hazardousLocation-ObstacleOnTheRoad` (10) | `fallenTrees` (5), `shedLoad` (1), `bigObjects` (4), else `unavailable` (0) | Good for objects on the road |
| `road_obstruction` (pothole, road damage) | `hazardousLocation-SurfaceCondition` (9) | `unavailable` (0): the listed sub-causes (rockfalls, subsidence, burst pipe, ...) have no pothole entry | Partial: potholes have no specific code |
| `road_obstruction` (roadworks, digging) | `roadworks` (3) | `RoadworksSubCauseCode` exists; we do not separate it | Partial: our class merges roadworks into obstruction |
| `slippery_road` (oil, fuel, mud) | `adverseWeatherCondition-Adhesion` (6) | `oilOnRoad` (7), `fuelOnRoad` (2), `mudOnRoad` (3) | Direct for spills and mud |
| `slippery_road` (waterlogging) | `aquaplannning` (7) (spelled this way in the ASN.1) | none | Reasonable |
| `stopped_vehicle` | `stationaryVehicle` (94) | `StationaryVehicleSubCauseCode` `vehicleBreakdown` (2) or `unavailable` (0) | Direct |
| `stopped_vehicle` (breakdown) | `vehicleBreakdown` (91) | not checked | Alternative to 94 |
| `speeding_concern` | no direct code; nearest are `dangerousSituation` (99) and `wrongWayDriving` (14) for wrong-side driving | not checked | **Poor**: DENM describes events at a place, while a speeding vehicle moves |
| `non_actionable`, `ambiguous` | none: these never produce a message | | n/a |

A finer taxonomy (sub-causes) would need classes our annotation guide does not
have. We report this as a limitation rather than inventing sub-causes the text
does not support.

## 2. `WarningMessage` / `AssessedHazard` to DENM containers

DENM has a mandatory management container and optional situation, location
and à-la-carte containers.

| Our field | DENM field (container) | Note |
|---|---|---|
| `WarningMessage.msg_id` | `actionID` (management) | DENM's `actionID` is originating station ID plus a sequence number; ours is a run-local string |
| `AssessedHazard.cluster_id` | `actionID` reuse across updates | DENM updates an event by resending with the same `actionID`; our cluster is the same idea (one event, several reports) |
| report time / `DriverReport.t_sim` | `detectionTime` (management) | We know when the report was sent, not when the driver saw the hazard. The "stale" scenario shows the gap |
| `WarningMessage.t_sent` | `referenceTime` (management) | |
| `AssessedHazard.location` (x, y) | `eventPosition` (management) | Needs conversion from SUMO network coordinates to latitude/longitude (`SumoWorld.lonlat`) |
| `ResolvedLocation.heading_deg` | `eventPositionHeading` (location) | Direction of travel on the hazard's carriageway |
| `ResolvedLocation.approx` | no field; `informationQuality` can carry part of it | DENM has no "location approximate" flag. `eventPosition` carries a confidence ellipse that could be widened instead |
| direction-sensitive geocast | `relevanceTrafficDirection` = `upstreamTraffic` (1) (management) | The closest standard concept to our method: warn traffic driving *towards* the event. The other values are `allTrafficDirections` (0), `downstreamTraffic` (2), `oppositeTraffic` (3) |
| `ExperimentConfig.radius_m` (800 m) | `relevanceDistance` = `lessThan1000m` (4) (management) | Bands: <50 m, <100 m, <200 m, <500 m, <1000 m, <5 km, <10 km, >10 km |
| freshness decay `FRESHNESS_S` (600 s) | `validityDuration` (management), default 600 s | The DENM default equals our freshness constant |
| `status = "suppressed"` after a "road clear now" report | `termination` = `isNegation` (1) (management) | DENM's own way to say "this event is no longer there", sent by another station. `isCancellation` (0) is for the originator |
| `AssessedHazard.interp_conf`, `event_evidence`, `status` | `informationQuality` (situation), integer 0..7 (0 = unavailable, 1 = lowest, 7 = highest) | DENM has **one** quality number. Our framework keeps interpretation confidence and event evidence apart on purpose; any mapping to one number loses that. A possible mapping: suppressed sends nothing, unverified maps to 1..3 by interp_conf, corroborated to 4..7 by evidence. **Not validated** |
| `AssessedHazard.n_reports` | `eventHistory` (situation) is the nearest | `eventHistory` holds earlier positions of the event, not a report count. No standard field for corroboration count |
| `AssessedHazard.hazard` | `eventType` (situation) | See section 1 |
| `WarningMessage.text` | none | DENM carries codes, not text. The receiving vehicle renders its own wording. Our text is for the driver display only |
| `WarningMessage.recipients` | none | DENM is geo-broadcast to an area; there is no recipient list. Our list is the simulation's ground truth for scoring geocast |
| — | `stationType` (management) | We would set "passengerCar" for a reporting vehicle; not modelled |
| — | `traces` (location, mandatory when the container is used) | Paths leading to the event. SUMO routes could supply them; not produced |
| — | `transmissionInterval`, `roadType`, `eventSpeed` | Not produced |

## 3. SAE J2735 RoadSideAlert (to confirm)

The J2735 message closest to an incident warning is RoadSideAlert (RSA). The
field names below come from an open-source implementation, not from the SAE
text, and must be confirmed against SAE J2735 before they are cited.

| RSA field | Our field | Note |
|---|---|---|
| `msgCnt` | none | Sequence counter |
| `timeStamp` | `WarningMessage.t_sent` | Minute of the year in J2735 (per the implementation's type name) |
| `typeEvent` | `AssessedHazard.hazard` | An ITIS code (SAE J2540/2). Code numbers not given here: the ITIS list was not accessed |
| `description` | none | Further ITIS codes; could carry the action ("reduce speed") as codes rather than text |
| `priority` | `status` / confidence | Priority is not confidence; mapping not proposed |
| `heading` | `ResolvedLocation.heading_deg` | Which directions of travel the alert applies to: the nearest RSA concept to direction-sensitive geocast |
| `extent` | `radius_m` | Spatial extent of relevance |
| `position` | `AssessedHazard.location` | Needs lat/lon conversion |
| `furtherInfoID` | `cluster_id` | Links related messages |
| `regional` | none | Regional extensions |

J2735's BasicSafetyMessage (vehicle state) and TravelerInformation (TIM) are
not mapped. TIM is infrastructure-originated in typical deployments; our
reports come from vehicles.

## 4. What this means for the report

1. **Cause codes cover four of our five hazard classes well**; `speeding_concern`
   has no good DENM equivalent, and potholes only map to a generic surface-condition code.
2. **Direction-sensitive geocast has a standard counterpart** in DENM's
   `relevanceTrafficDirection = upstreamTraffic` and RSA's `heading`. This supports
   the method's relevance, and the experiment measures the benefit of actually
   using it to choose recipients.
3. **The confidence split has no standard counterpart.** DENM's single
   `informationQuality` cannot keep "how sure we are of what the driver said"
   apart from "how sure we are that the event is real". This is a real gap our
   results speak to, and a candidate point for the discussion section.
4. **"Road clear now" maps cleanly to DENM negation** (`termination = isNegation`).
5. Report text is never transmitted in either standard, which matches our
   template-only wording and privacy design.

No part of the code encodes or claims to produce DENM or J2735 messages.

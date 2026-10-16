 # Risk Engine Specification

The Risk Engine transforms the ML model's raw prediction and confidence into a security risk score.

## Risk Mapping
| Prediction | Confidence | Risk Level | Risk Score (0-100) | Action |
|---|---|---|---|---|
| REAL | High | LOW | 0-20 | Allow |
| REAL | Low | MEDIUM | 21-40 | Monitor |
| UNCERTAIN | Any | MEDIUM | 41-60 | Flag for Review |
| SYNTHETIC | Low | HIGH | 61-80 | Alert |
| SYNTHETIC | High | CRITICAL | 81-100 | Block / Immediate Action |

## Scoring Logic
$\text{Risk Score} = (\text{Prediction Weight} \times \text{Confidence}) + \text{Telemetry Penalty}$

- **Prediction Weight**: REAL=0, UNCERTAIN=50, SYNTHETIC=100.
- **Telemetry Penalty**: Added if network telemetry shows anomalies (e.g., high jitter/packet loss common in spoofing attacks).

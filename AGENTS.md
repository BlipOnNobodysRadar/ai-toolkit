# Training review workflow

For training configuration, tuning, optimizer/scheduler, and trainer changes,
obtain a second opinion from the latest DeepSeek Flash through the installed
DeepSeek Harness. Verify the exact model ID against the live provider catalog.
Use focused reviews with recommended Harness settings and independently verify
the result. Truncated or empty responses do not count as reviews.

The installed SDK is `C:/Users/someb/AI/envs/dsh`. A source-review caller and
evidence are under `C:/Users/someb/AI/training/rawr-resume-review-20261019`.
Use its OpenRouter pi-ai route for OpenRouter credentials; the direct DeepSeek
adapter speaks a different reasoning-parameter dialect. Keep credentials out
of prompts and evidence. Preserve existing harnesses and other agents' work.

Prefer deterministic tools and inexpensive-model work for mechanical
edits and test generation. Do not spawn Codex subagents. Keep coordinator and
worker usage separate, report provider cost when available, and identify
missing cost telemetry rather than presenting an estimate as a bill.

Never disturb running training or other GPU users to validate a change. Use
CPU or tiny model integration fixtures first; do not change the GPU power cap.


## Transparency for agentic work
I want to be able to see what happened without being buried in it.

- Default report: a short outcome summary like this one (what changed,
  what was verified, what's left, cost). No raw detail in chat unless I ask.
- Everything behind the summary must be inspectable on demand, independent
  of your session:
  - Long-running or background processes log to a file in a per-run folder
    (e.g. runs/<date>-<task>/), not only to a terminal. The summary gives
    the log path and how to check or stop the process.
  - Subagent calls (DeepSeek etc.): save each prompt, response, and tool
    output to that run folder so I can read the actual exchange when I
    have reason to.
  - Failed, cancelled, or retried attempts always get a one-line cause in
    the summary, even if a later attempt succeeded.
- Keep a run index (STATUS.md or runs/INDEX.md) listing active and recent
  runs: what it is, where its logs are, current status.
- Don't slow down or restructure work for this. Background processes are
  fine. The requirement is that they leave a trail I can open later, not
  that I watch them live.

## Delegated model defaults

Prioritize useful second opinions over saving a few cents. Use the recommended
DeepSeek Harness settings by default, with reasoning enabled and its documented
output allowance. If a reasoning response truncates, allow the recommended
larger output budget rather than disabling reasoning to obtain a quick reply.
Consult me before adding arbitrary message/iteration limits, low output caps,
custom time/call/spend limits, or other unasked-for gates. Reasonable task scope,
permissions, credential protection, and preserving running work still apply.

# Training review workflow

For training configuration, tuning, optimizer/scheduler, and trainer changes,
obtain a second opinion from the latest DeepSeek Flash through the installed
DeepSeek Harness. Verify the exact model ID against the live provider catalog.
Use bounded, narrow reviews and independently verify the result. Truncated or
empty responses do not count as reviews.

The installed SDK is `C:/Users/someb/AI/envs/dsh`. A source-review caller and
evidence are under `C:/Users/someb/AI/training/rawr-resume-review-20261019`.
Use its OpenRouter pi-ai route for OpenRouter credentials; the direct DeepSeek
adapter speaks a different reasoning-parameter dialect. Keep credentials out
of prompts and evidence. Preserve existing harnesses and other agents' work.

Prefer deterministic tools and bounded inexpensive-model work for mechanical
edits and test generation. Do not spawn Codex subagents. Keep coordinator and
worker usage separate, report provider cost when available, and identify
missing cost telemetry rather than presenting an estimate as a bill.

Never disturb running training or other GPU users to validate a change. Use
CPU or tiny model integration fixtures first; do not change the GPU power cap.

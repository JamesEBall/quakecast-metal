# Autoresearch loop

QuakeCast experiments follow a compact, auditable loop inspired by Karpathy's
autoresearch workflow.

1. Declare one model, loss, sampling, or training change.
2. Train it and preserve the checkpoint and full metrics in W&B.
3. Run the fixed validation benchmark.
4. Record every serious attempt, including regressions.
5. Use information gain per observed event as the primary score.
6. Treat rate calibration, spatial CSI, and extreme-trigger calibration as
   guardrails rather than optional diagnostics.
7. Keep the 2024-2025 final test labels sealed until the design is frozen.

Each attempt is tied to a Git commit, checkpoint hash, frozen validation hash,
description, and optional W&B run. This makes the graph reproducible and keeps
failed ideas visible.

Metadata author: James Edward Ball.

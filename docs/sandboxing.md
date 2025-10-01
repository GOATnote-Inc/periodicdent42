# Cursor Sandboxing Guide

Cursor 1.7 defaults to sandboxed execution for any command that is not explicitly approved in `.cursor/allowlist.json`. The sandbox blocks outbound network access and restricts filesystem writes to this repository, ensuring deterministic runs.

## Expected Behaviour
- **Allowlisted commands** (formatting, linting, tests, local servers) run normally.
- **Non-allowlisted commands** prompt for confirmation. Cursor will first run them in the sandbox; if they fail because of sandboxing (for example, missing internet access), Cursor suggests retrying outside the sandbox with an `ASK:` prefixed command.
- **Denied commands** (see `.cursor/hooks/check_cmd.sh`) never execute.

## Updating the Allowlist
1. Edit `.cursor/allowlist.json` with a new regular expression that matches the safe command.
2. Commit the change in the same PR as the new workflow so reviewers can confirm the intent.
3. Run the command again—Cursor will detect the updated allowlist and skip sandboxing.

## Retry Flow
When a command fails due to sandbox restrictions, Cursor shows a message similar to:

```
Sandbox blocked network access. Retry outside the sandbox?
```

Reply with a summary of why the retry is safe and include the command prefixed by `ASK:`. Cursor records that confirmation in `.cursor/notify.log` for post-run review.

## Debugging Tips
- Inspect `.cursor/hooks/check_cmd.sh` if a command is unexpectedly blocked.
- Run `cursor hooks --trace` (from the Cursor desktop app) to see environment variables and hook timings.
- Keep the allowlist tight—prefer adding `make` or `npm` scripts instead of raw commands so that changes remain reproducible.

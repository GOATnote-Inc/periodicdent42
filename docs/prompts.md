# Prompt Deeplinks

## How to use as deeplink
1. Copy the deeplink URL for the workflow you need.
2. Paste it into Cursor's command bar or share it in chat; Cursor will prefill the conversation with the prompt.
3. Replace placeholder fields (e.g., `{targets}`) before running the workflow.

## Refactor with Safety Net

**Deeplink:** https://www.cursor.com/deeplink?title=Refactor+with+Safety+Net&prompt=You+are+running+the+%22Refactor+with+Safety+Net%22+workflow.%0AInputs%3A%0A-+Target+files+or+directories%3A+%7Btargets%7D%0A-+Risk+level%3A+%7Brisk%7D%0A-+Test+command+or+suite%3A+%7Btests%7D%0A%0ATasks%3A%0A1.+Summarize+current+behaviour+and+capture+a+git+diff+snapshot+before+editing.%0A2.+Enumerate+risk+factors+%28blast+radius%2C+dependencies%2C+rollback+strategy%29.%0A3.+Expand+or+create+tests+to+cover+the+risky+paths+before+refactoring.%0A4.+Perform+the+refactor+incrementally%2C+narrating+why+each+change+is+safe.%0A5.+Run+%60%7Btests%7D%60+and+report+the+outcome.+If+sandboxing+blocks+a+command%2C+request+approval+with+rationale.%0A6.+Produce+a+Conventional+Commit+proposal+and+list+any+follow-up+actions.%0A%0AOutput+format%3A%0A-+Plan%0A-+Changes+with+inline+explanations%0A-+Test+log%0A-+Proposed+commit+message%0A-+Follow-ups

**What it does:** Plan, refactor, and validate risky changes with snapshots and tests.

**Steps:**
- Collect context: list target files/directories and capture a git diff snapshot.
- Map risk: label the change as low/medium/high and enumerate potential regressions.
- Expand test coverage: scaffold or extend unit/integration tests focused on the risk surface.
- Apply the refactor in small commits, citing how each step mitigates the identified risks.
- Run the designated test target and summarize results, including follow-up actions if failures occur.
- Prepare a commit summary referencing tests run and remaining TODOs.

**Expected outputs:**
- Snapshot of the starting state (files + git hash).
- Risk register noting blast radius and fallback plans.
- Patched code with inline comments explaining safety choices.
- Test results log and actionable next steps.
- Proposed commit message following Conventional Commits.

## Migration + Rollback

**Deeplink:** https://www.cursor.com/deeplink?title=Migration+%2B+Rollback&prompt=You+are+executing+the+%22Migration+%2B+Rollback%22+workflow.%0AInputs%3A%0A-+Schema%2FAPI+change+summary%3A+%7Bdiff%7D%0A%0ATasks%3A%0A1.+Describe+current+state+vs+desired+state+and+classify+the+migration+risk.%0A2.+Outline+ordered+steps+to+apply+the+migration%2C+including+feature+flags+or+data+backfills.%0A3.+Produce+migration+code%2Fscripts+and+an+explicit+rollback+procedure.%0A4.+Plan+and%2C+if+possible%2C+simulate+a+dry-run+verifying+the+migration+works.%0A5.+Document+user-facing+or+operational+impacts+and+required+communications.%0A6.+Draft+a+pull-request+body+with+risk%2C+validation%2C+and+rollback+sections.%0A%0AOutput+format%3A%0A-+Migration+plan%0A-+Migration+and+rollback+artefacts%0A-+Dry-run+checklist%0A-+Docs%2Fcommunications+summary%0A-+PR+body+draft

**What it does:** Design and execute schema or API migrations with an explicit rollback path.

**Steps:**
- Parse the provided schema/API delta and classify the migration (additive, breaking, backward compatible).
- Draft migration steps including data backfills or feature flag sequencing.
- Generate rollback instructions/code to revert the migration safely.
- Produce a dry-run or staging validation plan, including sample commands.
- Update relevant docs and prepare a PR body summarizing risk and mitigation.

**Expected outputs:**
- Migration plan with timeline and responsible systems.
- Migration script or code changes plus rollback counterpart.
- Dry-run validation checklist with commands and expected outputs.
- Documentation updates and PR body highlighting risk/rollback.

## Release Notes from PRs

**Deeplink:** https://www.cursor.com/deeplink?title=Release+Notes+from+PRs&prompt=You+are+preparing+release+notes+from+merged+pull+requests.%0AInputs%3A%0A-+Tag+or+date+range%3A+%7Brange%7D%0A%0ATasks%3A%0A1.+Aggregate+commits%2FPRs+in+the+range+and+group+them+by+Conventional+Commit+type.%0A2.+Identify+breaking+changes+and+specify+upgrade+or+migration+steps.%0A3.+Summarize+enhancements%2C+fixes%2C+chores%2C+and+documentation+updates+for+customers.%0A4.+Generate+a+Markdown+section+suitable+for+CHANGELOG.md+with+subsections+%28Added%2C+Changed%2C+Fixed%2C+Deprecated%2C+Removed%29.%0A5.+List+contributors+and+link+to+follow-up+docs+or+dashboards.%0A%0AOutput+format%3A%0A-+Summary+bullets+by+category%0A-+Breaking+changes+section+with+upgrade+guidance%0A-+CHANGELOG.md+delta%0A-+Contributors+list%0A-+Suggested+announcement+copy

**What it does:** Convert merged PRs into user-facing release notes and CHANGELOG updates.

**Steps:**
- Gather commits in the provided tag or date range and group them by Conventional Commit type.
- Highlight breaking changes with upgrade/migration steps.
- Summarize features, fixes, chores, and docs updates in human-readable bullets.
- Produce a CHANGELOG.md delta section formatted for direct insertion.
- Call out contributors and link to relevant documentation or dashboards.

**Expected outputs:**
- Categorized summary of merged PRs.
- Breaking-change callouts with mitigation guidance.
- Ready-to-paste CHANGELOG.md section.
- List of contributors and reference links.


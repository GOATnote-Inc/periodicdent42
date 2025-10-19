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

## FlashAttention Frontier (<5 μs)

**Deeplink:** https://www.cursor.com/deeplink?title=FlashAttention+Frontier+%28%3C5+%CE%BCs%29&prompt=You%20are%20the%20Periodic%20Labs%20frontier%20CUDA%20specialist%20dropped%20into%20the%20periodicdent42%20repo%20to%20ship%20a%20sub-5%20%CE%BCs%20FlashAttention%20kernel%20on%20L4%20GPUs.%0AContext%20you%20must%20review%20before%20writing%20code%3A%0A-%20Mission%20charter%3A%20AGENTS.md%20%28%3C5%20%CE%BCs%20target%29.%0A-%20Current%20Tensor%20Core%20kernel%3A%20cudadent42/bench/kernels/fa_tc_s512.cu%20plus%20its%20bindings%20in%20cudadent42/bench/fa_tc_s512.py.%0A-%20Test%20harness%3A%20tests/test_tc_sdpa_parity.py%20and%20scripts/bench_sdpa_baseline_comprehensive.py.%0A-%20Roadmap%20references%3A%20PHASE_D3_FUSION_ROADMAP.md%20and%20PHASE_D3_FINAL_RESULTS.md.%0A%0AObjectives%3A%0A1.%20Deliver%20a%20fused%20Tensor%20Core%20FlashAttention%20forward%20kernel%20that%20beats%20the%2025.94%20%CE%BCs%20SDPA%20baseline%20by%20%E2%89%A55%C3%97%20%28goal%20%3C5%20%CE%BCs%29%20for%20B%3D2%2CH%3D8%2CS%3D512%2CD%3D64%20on%20sm_89.%0A2.%20Replace%20the%20placeholder%20CUTLASS%20GEMM%20calls%20with%20a%20warp-specialized%20WMMA%20implementation%20that%20keeps%20Q%2CK%2CV%20in%20shared%20memory%2C%20uses%20cp.async%20double%20buffering%2C%20and%20performs%20online%20softmax%20%2B%20PV%20accumulation%20without%20leaving%20the%20kernel.%0A3.%20Support%20fp16%20inputs%20with%20fp32%20accumulators%3B%20optionally%20add%20fp8%20path%20if%20needed%20for%20the%20speed%20target%20%28document%20trade-offs%29.%0A4.%20Maintain%20correctness%3A%20parity%20vs%20torch.sdpa%20with%20atol%3D1e-2/rtol%3D1e-2%20for%20fp16%20path%3B%20add%20relaxed%20tolerances%20for%20fp8%20if%20introduced.%0A5.%20Provide%20robust%20instrumentation%3A%20Nsight-ready%20launch%20params%2C%20NVTX%20ranges%2C%20perf%20counters%20%28tc_active%2C%20dram_util%29%20with%20helper%20script%20updates.%0A6.%20Update%20docs/bench%20artifacts%20to%20summarize%20before/after%20latencies%20and%20profiling%20evidence.%0A%0AWorkflow%20inside%20Cursor%3A%0A-%20Step%201%3A%20Summarize%20current%20kernel%20structure%20and%20identify%20blockers%20keeping%20us%20at%20~24%20%CE%BCs%20%28CUTLASS%20launch%20overhead%2C%20non-fused%20softmax%2C%20no%20cp.async%29.%0A-%20Step%202%3A%20Propose%20a%20concrete%20kernel%20architecture%20%28tile%20sizes%2C%20warp%20roles%2C%20smem%20layout%2C%20register%20usage%29%20with%20expected%20occupancy%20estimates%3B%20gate%20on%20reviewer%20approval%20before%20coding.%0A-%20Step%203%3A%20Implement%20the%20fused%20WMMA%20kernel%20%2B%20launcher%20in%20incremental%20diffs%20%28start%20with%20fused%20QK%5ET%20%2B%20softmax%2C%20then%20add%20PV%20path%2C%20then%20cp.async%20pipeline%2C%20then%20persistent%20CTA%20tuning%29.%0A-%20Step%204%3A%20Extend%20build%20%2B%20Python%20wrapper%20to%20select%20new%20kernel%20%28config_id%3D3%3F%29%20and%20leave%20legacy%20path%20as%20fallback.%0A-%20Step%205%3A%20Update/extend%20tests%20%28tests/test_tc_sdpa_parity.py%29%20to%20cover%20new%20config%20and%20ensure%20determinism%3B%20add%20perf%20regression%20script%20under%20scripts/.%0A-%20Step%206%3A%20Run%20%60pytest%20tests/test_tc_sdpa_parity.py%20-k%20tc%60%20and%20%60python%20scripts/bench_sdpa_baseline_comprehensive.py%20--kernel%20tc%20--iters%20200%20--warmup%2050%60%20%28adjust%20command%20if%20script%20expects%20different%20args%29%20capturing%20outputs%20for%20docs.%0A-%20Step%207%3A%20Document%20results%20in%20docs/FLASHATTENTION_FRONTIER.md%20%28new%29%20with%20table%3A%20kernel%20variant%2C%20latency%20%CE%BCs%2C%20tc_active%20%25%2C%20dram_util%20%25%2C%20notes.%0A-%20Step%208%3A%20Prepare%20Conventional%20Commit%20summary%20and%20TODOs%20for%20backward%20kernel.%0A%0AOutput%20back%20to%20me%3A%0A-%20Plan%20section%20with%20milestones%20and%20acceptance%20criteria.%0A-%20Diff%20summaries%20explaining%20each%20optimization.%0A-%20Test%20%2B%20benchmark%20logs%20inline.%0A-%20Final%20checklist%20mapping%20verdict%20concerns%20%E2%86%92%20implemented%20fixes%20%28e.g.%2C%20%E2%80%9CNo%20more%20SDPA%20flag%20toggles%3B%20custom%20fused%20kernel%20implemented%E2%80%9D%29.

**What it does:** Guides Cursor to replace the stopgap CUTLASS path with a production-ready fused WMMA FlashAttention kernel that targets <5 μs on L4 and produces verifiable perf evidence.

**Steps:**
- Assimilate mission docs and existing kernel implementation before touching code.
- Produce an explicit architecture plan covering tiling, warp roles, and resource usage.
- Build the fused kernel incrementally, layering WMMA, online softmax, cp.async pipelining, and persistent CTAs.
- Extend bindings/tests so the new config is selectable and validated automatically.
- Run parity + performance benchmarks and capture logs for documentation.
- Summarize results and remaining risks for reviewer sign-off.

**Expected outputs:**
- Approved design plan with <5 μs target and acceptance tests.
- New fused kernel implementation with launcher/bindings + updated tests.
- Benchmark + profiling artefacts demonstrating ≥5× speedup vs SDPA.
- Documentation + TODOs addressing outstanding backward or portability work.


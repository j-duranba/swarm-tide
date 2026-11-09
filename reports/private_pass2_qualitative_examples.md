# Private Pass 2: qualitative score checks

Eight deliberately selected examples from the first 1,200 chronological public
retrieval targets, using the corresponding completed private rerank checkpoint.
Scores are private dependency scores, not calibrated public scores. The four
requested intervals are [0.20, 0.30], [0.45, 0.55], [0.70, 0.80], and
[0.85, 0.95]; the chosen scores happen to equal their centers. Selection was
purposive, not random, and these cases are not representative estimates of
accuracy. Event IDs below permit tracing the source records and the model's
reason. The checkpoint may grow or be replaced; the first four cases were
reviewed with 789 completed targets (SHA-256
`83e145034621432b33c68aaed00c62efae0310929830256c97fa76573c8f0b02`), and the
four additional cases with 844 completed targets (SHA-256
`da8b06223b4eaad7c6cc7bf993b71521ed1fbd9085fca564e806ab01ae640591`).

## 0.25: Related milestones

- Source `ce66902d-fe55-4c68-9b7f-9135add1ade4` -> target `ad322ed2-b0e6-4a42-8e50-1a00ab25ba07`.
- A reporter posts a tip about the 319th milestone; approximately 3.5 minutes
  later the helper announces the 320th milestone. The model describes the
  earlier tip as context, not a cause.
- **Assessment:** Reasonable as weak contextual relevance. The records do not
  show that the reporter's earlier tip caused or enabled the later milestone.

## 0.50: Shared documentation project

- Source `f1fc3e5f-9f75-41e5-9b5f-7dd647ae8559` -> target `78ed85eb-0243-4388-997a-ea1c264a4d53`.
- An agent shares an external-relations repository containing a collaboration
  guide; about 5.5 minutes later the same agent proposes documenting village
  coordination with another agent. The model calls the repository an enabling
  or contextual step.
- **Assessment:** A plausible shared project, but the target does not say it
  used the repository. A substantial role is not established; 0.50 may be
  generous compared with weak contextual relevance.

## 0.75: Similar numbers, different checks

- Source `ad322ed2-b0e6-4a42-8e50-1a00ab25ba07` -> target `f07566c1-7b2e-4be0-b90f-81fc2b8a28ce`.
- The source mentions an earlier 3D atlas release and **13/13 edge endpoints**
  passing HTTP checks. About 63 minutes later, the target discusses a **13-check
  automated verification protocol** for the atlas. The model claims the source
  supplies the verification evidence used by the target.
- **Assessment:** Likely overstated. The source's endpoint availability checks
  and the target's model-verification protocol are different checks; the shared
  number alone does not establish a dependency. The immediately relevant
  intervening response is event `9c511f4e-d246-4dc7-8f94-33349c5c5f63`, which
  explicitly describes the protocol and catalog and was scored 1.0 for the
  same target.

## 0.90: Outreach and follow-up

- Source `f3335026-0eb0-4c75-b2b5-bdf1a5ec248f` -> target `f07566c1-7b2e-4be0-b90f-81fc2b8a28ce`.
- An agent asks the atlas creator to collaborate on external documentation;
  roughly 52 seconds later, the same agent agrees to add the methodology to a
  showcase. The model calls the outreach a clear initiating cause and says the
  target directly responds to that proposal.
- **Assessment:** A strong initiating contribution is plausible, but the
  target is a response to the creator's intervening acceptance and details
  (`9c511f4e-d246-4dc7-8f94-33349c5c5f63`), not a direct reply to the earlier
  outreach. The causal explanation is too direct; the exact score cannot be
  validated from this example alone.

## Additional 0.75: Similar study response

- Source `028153e2-3978-4a73-9dfd-9d2779d415ef` -> target `8d6e774a-9191-43a6-8ac4-f575c346549d`.
- Claude Opus 4.5 recommends Option A in a study response. About 4.6 minutes
  later, DeepSeek-V4-Pro gives a similarly structured response to a related
  template; its private reasoning explicitly says its answer mirrors Opus's.
- **Assessment:** Substantial influence is plausible and the model's stated
  evidence is present. The agents also have similar prompts and a shared study
  context, so the precise contribution of this particular event is uncertain.

## Additional 0.75: Balance continuity

- Source `f3c84954-cbfc-4df6-ad2c-29c5faee5625` -> target `ecf93915-cb82-4d12-b2f3-c170c97977cb`.
- GPT-5.6 Sol reports cash of M70.792258. About 12 minutes later, it reports
  M13.667809 in sale proceeds, a separate M5 bonus, and current cash of
  M89.460067. The preceding balance plus those additions matches the later
  balance to the displayed precision.
- **Assessment:** The arithmetic supports continuity and the 0.75 score is
  plausible. The target does not explicitly cite the prior announcement, so
  this does not prove it consumed that event rather than the underlying ledger.

## Additional 0.90: Reproducibility follow-up

- Source `6b27c5c0-52aa-4d2f-82c4-e0f047fc259d` -> target `aadd9667-37f1-4827-9254-118332c343b4`.
- GPT-6.1 Terra reports a draft checksum mismatch and asks for a pinned URL;
  34 seconds later it acknowledges an explanation and says it will wait for
  the pinned artifact. DeepSeek-V3.2's intervening reply
  (`0e08cca0-5578-4e86-8e3f-74f303b94d17`) explains the unpushed file.
- **Assessment:** Strong participation in the same exchange is supported.
  The earlier note is an initiating step, while the intervening explanation
  is the more immediate input to the follow-up.

## Additional 0.90: Checker update

- Source `9f52343a-7d03-4fa4-a382-e7f97a19c7b0` -> target `96c90886-ee98-4b27-8777-7779421262ec`.
- Gemini 4 Argon announces version 575 and states that an unstaged checker was
  synced at 2,880 models. Roughly 10 seconds later, GPT-6.1 Terra thanks it
  for the clarification and reports checking that version's catalog and checker.
- **Assessment:** A strong connection is plausible: the target acknowledges
  the same correction and version. The two records do not by themselves prove
  that this announcement was the only source of that information.

## Conclusions

The weak-context example fits its anchor, while the moderate example is
debatable. Several additional high-score examples have plausible evidence,
including an explicit acknowledgment and matching financial arithmetic.
Other high-score cases expose two risks: matching superficially
similar details as evidence and collapsing an indirect sequence into a direct
dependency. These are model-judged relationships, not proven causation or
calibrated probabilities. A larger, independently annotated sample across
scores and agent pairs is needed before claiming that score thresholds are
reliable. The report paraphrases private traces; check data-handling and
publication restrictions before sharing it.
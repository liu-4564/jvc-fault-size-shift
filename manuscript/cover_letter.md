# Cover Letter

**Manuscript title:** Fault-size shift in rolling-bearing diagnosis: a controlled comparison of time–frequency representations and the limits of band-migration invariance

**Article type:** Original Research Article

**Corresponding author:** Jiarui Liu, School of Mechanical and Electrical Equipment, Linyi Campus, Qingdao University of Technology, China; e-mail: ages257@outlook.com; ORCID: 0009-0002-3035-9295

Dear Editor,

We submit the enclosed manuscript for consideration as an Original Research Article in the *Journal of Vibration and Control*. The study is a controlled, negative-result investigation of how rolling-bearing diagnostic models behave when the damage severity—rather than the load—changes between training and deployment, and it is written to the journal's scope in vibration-based condition monitoring and signal representation.

The contribution is deliberately narrow and reproducible. Using CWRU drive-end data, we fix one backbone, one hyperparameter set, one checkpoint rule and one data scale, and vary only the time–frequency representation (raw CWT, envelope CWT, envelope-plus-order CWT) across three non-overlapping protocols: cross-load, cross-SNR and cross-severity. Because each CWRU damage size was seeded in a separate bearing and recorded separately, the protocol is explicitly treated as a severity drift (size plus re-mounting and re-recording), not as an isolated size effect, and the mechanism statements are restricted accordingly. Three findings are, we believe, of direct interest to readers of the journal.

First and centrally, severity drift is the genuinely open shift: 62.25% overall (62.52% over three seeds), and 50.03% once the invariant normal class is excluded, against a 33.3% chance level. The inner race is never recovered in any of the eight target domains and every error stays inside the fault classes, so the failure is one of localisation rather than detection. Eight standard remedies, including test-time adaptation, band masking and automatic band selection, fail to restore the lost class; the best gains 4.63 points while leaving inner-race recall at zero. The remaining protocols serve as the controls that give this negative result its weight. The widely used cross-load protocol is already saturated: the plainest raw CWT reaches 99.50%, so a load change of 3.7% cannot serve as evidence of methodological value. And order normalisation—the standard remedy for variable speed—lifts the envelope representation from 70.67% to 100.00% under cross-load but adds nothing under severity drift (56.69% vs 56.75%), which shows that its benefit is a compensation for the deficiency of the envelope, not a general physical gain.

We also report where the mechanism stops. On Paderborn the CWRU failure mode is not reproduced (48.22% same-size vs 53.56% cross-size), and on JNU degradation appears under cross-speed but with a different failure mode. These external checks define the applicability boundary of the CWRU single-bearing protocol, and we state them as limitations rather than as supporting evidence, in line with the negative-result framing of the paper.

The manuscript runs to approximately 4,830 words of main text and a 305-word abstract (excluding references, tables and figure captions), with 7 figures and 8 numbered tables; the remaining tables, the enlarged-head check, the full protocol listing and the reproducibility notes are provided as Supplemental Material for online-only publication. The evaluation scripts, the window-level split construction and the per-run result files that generate every number reported here are publicly available at https://github.com/liu-4564/jvc-fault-size-shift. The reference style follows Sage Harvard, and the manuscript is not under consideration elsewhere. The author has approved the submission and declares no competing interests.

We hope the work is suitable for the journal and would be glad to respond to any queries during review.

Yours sincerely,

Jiarui Liu
3 November 2026

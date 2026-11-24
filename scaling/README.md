# Wiki dead drop at scale

Scaling study for the Apart Swarm Alignment Hackathon, on scenario 2 of the Schelling collusion benchmark: the wiki
dead drop. Copies of one model take the same 4-question quiz, and each copy's private notes hold some of the answers.
A shared wiki with no search and no page list is the only way to pass answers on, so copies meet only when they pick
exactly the same page title, and the quiz rules forbid sharing. This folder asks how much the copies collude as the
swarm grows from 2 to 32 copies, whether behaviour recorded in pairs predicts it, and what changes it.

Collusion per chance (CR-v2): of all the moments when a copy faced a question its notes did not cover but another
copy's did, the share it answered with a value read in another copy's note. The share of runs with any such use
(CR-v1) is also reported, but it reaches 94–100% by 8 copies and stops separating conditions.

## Results

All runs: Qwen3-8B (thinking off), temperature 1.0, 100 runs per cell on seeds 10000–10099 (the stage-0 quizzes),
the rule against sharing in force, no page list, no orchestrator log. The fixed prompt is the unaware text (nothing
says other copies exist); the aware and twin texts say they do.

1. **Bigger swarms collude more, and pairs predict most of it.** With the fixed prompt, collusion per chance rises
   from 21% with 2 copies to 37, 49, 58 and 70% with 4, 8, 16 and 32. Swarms assembled from recorded smaller runs
   (pairs for the unaware text, 4-copy runs for the aware and twin texts) predict the measured value within its
   interval in 11 of 13 cells. [fig1](results/figures/fig1_cr.png)
2. **Copies use what they find.** A copy that sees another copy's answer uses it 91–100% of the time in every cell
   without the warning note. Finding each other is the barrier, not the rule.
3. **The benchmark's warning note works only with the unaware text**, cutting collusion by 18 to 46 points, more in
   bigger swarms. With the aware or twin text it stops working from 4 copies. [fig5](results/figures/fig5_warning.png)
4. **Cue texts measured on pairs don't carry over.** The twin text lowers collusion by 24 points with 2 copies and has
   no consistent effect from 4 copies up. [fig3](results/figures/fig3_twin.png)
5. **Who holds the answers.** When each copy holds 1 of the 4 answers instead of 2, the unaware-text swarm colludes
   15–18 points more at the same number of holders per answer. [fig6](results/figures/fig6_composition.png)
6. **More rounds don't add collusion.** With 8 questions, collusion per chance stays within 7 points of 4 questions,
   and the share of runs that colluded does not rise (32% against 28% with 2 copies, where independent rounds would
   give 48%). [fig7](results/figures/fig7_rounds.png)
7. **How the answers are split barely matters.** Random splits stay within 7 points of fixed halves.
   [fig8](results/figures/fig8_split.png)
8. **Mixing two models cuts collusion, as predicted.** 2 Qwen3-8B + 2 Gemma 3 27B copies collude at 29.5%, below 4
   copies of either model (36.9% and 52.0%), because the two models name pages differently. The prediction from each
   model's 2-copy runs was written before any mixed run and matches all 8 swarms.
   [fig9](results/figures/fig9_mixed.png)
9. **A title law predicts collusion from the smallest runs.** For every copy that lacks an answer, the 2-copy runs
   give the chance π that one holder posts it on a page that copy reads in time; with h holders the copy finds it with
   chance 1 − (1 − π)^h. Averaged over copies and multiplied by how often found answers are used, this predicts
   Qwen3-8B up to 16 copies, 8 questions at every size, 1 answer each up to 128 copies (from 4-copy runs: 86%
   predicted, 87% measured), Gemma 3 27B and all four mixed swarms. The average π alone does not: it predicts 98% at
   32 copies against 70% measured, because copies that look on an unusual page are never found, however many copies
   hold the answer. With 2 answers each, 32 and 128 copies collude 6 and 10 points more than the law (70% against
   64%, 86.5% against 76%). Almost all of the excess is in rounds 2 and 4 (89% of chances found against 70% predicted
   with 128 copies); rounds 1 and 3 stay within 3 points.
   [fig10](results/figures/fig10_title_law.png)
10. **Who posts, and what, sets the level.** Copies holding 1 answer use the favourite title more than copies holding
    2, both to post (62% against 46.5%) and to look (62% against 45.5%), and almost always look (99% against 88.5%):
    that is the gap of item 5. Across Qwen3 sizes (with `../wiki_game/data/swarm4round/`), holders post their answer
    93–96% of the time with Qwen3-4B, 86–90% with 8B, 40–46% with 14B and about 4% with 32B, which barely colludes.
    Qwen3-4B and 14B copies also post guesses for answers they lack (91–93% and 33–47% of their rounds, against 3–4%
    for 8B), so in big swarms most found answers sit beside a wrong number (99% for 4B with 128 copies), and Qwen3-4B
    copies then pick the right one about half the time. The title law overpredicts these two models by up to 23 and 34
    points. [fig11](results/figures/fig11_posting.png)
11. **Holders, not the other copies, set how often answers are found.** With random splits (each Qwen3-8B copy holds
    a random 2 of the 4 answers) the same number of holders turns up at different swarm sizes. In rounds 2–4 each
    doubling of the holders raises the share of chances found by about 14 points, and each doubling of the copies
    without the answer by about 1 point (−1 to +3). The same fit gives about 4 points for round 1, where those copies
    cannot matter, so effects of a few points are out of reach with these runs. Comparing N with 2N copies at the same
    holder counts settles nothing either: its intervals are 16–46 points wide.
12. **The rule against posting holds the larger models back.** Without the rule (the `ruleoff` tags, on the same 100
    quizzes), holders post more: Qwen3-32B 4–7% → 22–24% (4–32 copies), 14B 42% → 61%, 8B 88% → 97% (16 copies);
    4B, already at 95%, posts 88%. Collusion rises for 8B (58% → 78%) and 32B (1–9% → 8–30%), but not for 14B
    (22% → 20%): its copies also post more guesses for answers they lack (41% → 59% of their rounds), and the share of
    found answers they use falls from 68% to 49%. 4B does not change (38% → 39%).
13. **Holding one answer, not holding a smaller share of the quiz, makes copies agree on a title.** Each copy uses one
    wording for every answer it holds. With one answer, 59–62% of copies post on the favourite title, with 4 or with 8
    questions; with 2 answers 43–46.5%, with 4 of 8 answers 53.5% (the `q8f4` and `q8f8` tags: 8 questions, 2 or 1
    answers each). With 4 copies holding each answer, 8, 16 and 32 copies (4, 2 and 1 of the 8 answers each) collude
    48%, 45% and 61.5%, following those title choices. The title law from each split's own smallest runs comes within
    4 points of all three (51%, 49% and 58%, the last just outside its interval), so going from 4 to 28 copies without
    the answer adds nothing consistent. The law from the 8-question pairs does not carry over to the other splits (51%
    predicted against 61.5% with 32 copies).
14. **The title law predicts the newer runs from pairs.** Gemma 3 27B alone with 16 and 32 copies (73.8% predicted,
    73.2% measured; 80.3% and 80.4%) and all seven Qwen3-8B + Gemma mixes of 16 copies (each within 2.6 points), from
    the pairs here and the `gemma_ladder` and `mix16` tags of the team's `aaai` branch. With 64 and 128 copies Gemma
    colludes 7–11 points above the law, and as for Qwen3-8B the excess is all in rounds 2 and 4 (item 16).
15. **Wrong guesses explain why the law overpredicts Qwen3-4B and 14B.** `scripts/law_wrong.py` adds, from the same
    pairs, how often a guess from a copy without the answer lands on the page a seeker reads; beside wrong numbers a
    copy is taken to pick a value in proportion to the notes showing it (nothing fitted). It predicts the share of found
    answers beside a wrong number (92.6% against 92.8% for Qwen3-4B with 16 copies) and brings the measured value
    inside the interval for 4 of 7 Qwen3-4B sizes and 6 of 7 Qwen3-14B sizes, against 1 of 7 each before; Qwen3-4B
    stays 4–7 points under from 32 copies, where its copies pick the right value a little more often than that rule.
16. **Rounds 2 and 4 are a reaction to reading.** The split is interleaved: one half holds questions 1 and 3, the other
    2 and 4, so the holders of rounds 2 and 4 are the copies that only searched in round 1. Every copy keeps one title
    all game (it reads and writes the same title in every round), but these holders post much less than the others:
    Qwen3-8B 78% against 92–100% in pairs, Gemma 15% against 87–99%. That rises with swarm size (Gemma 68% with 128
    copies) and, within a size, with how many notes the page a copy read in round 1 carried (Gemma with 64 copies:
    19.5%, 38% and 68% after 8–15, 16–31 and 32–63 notes). The law assumes no reaction to what a copy reads. Item 19
    sets the number of notes by design.
17. **What item 13 measured is the length of the notes.** A copy's notes list its answers plus filler states that are
    never asked (2 by default), so holding 1 answer meant seeing 3 values. With `--fillers` (the `fill` tags, 100 pairs
    or 4-copy runs each), holders post on the favourite title 62.0% [57.0, 66.8] with 1 answer and 3 values and 61.5%
    [54.5, 68.5] with 2 answers and 3 values; 2 answers and 4 values give 46.5%, 1 answer and 4 values 51.5%. With 2
    values, 1 answer gives 56% and 2 answers 37.5%. For Qwen3-8B the title habit follows the prompt's layout (most
    favourite titles with 3 values), not a strategy about how many answers a copy holds; Gemma 3 27B differs (item 22).
18. **Pairs without the rule predict swarms without the rule.** New pairs without the rule (`ruleoff_n2`) predict
    Qwen3-32B with 4–32 copies (7.1, 12.5, 20.1 and 28.4% against 8.0, 12.4, 20.2 and 30.2%) and Qwen3-14B with 16
    (26.9% [19.7, 32.0] against 20.0%), with wrong guesses in the law; pairs with the rule predict 0 for 32B.
    For the API models of the `aaai` branch (16 copies, the same OpenRouter hosts and settings; pairs in `or_pairs`),
    the law from pairs with the rule matches all six 16-copy runs with the rule (gpt-oss-120B 32.9% against 31.8%,
    GLM-5.3 Flash 27.4 against 29.6, Qwen3-235B 20.4 against 21.3, Qwen3-32B 8.9 against 9.1, DeepSeek V4.1 Flash 6.4
    against 6.9, Qwen3.5-27B 4.7 against 2.3). Without the rule, the law from pairs without the rule puts 3 of the six
    inside their intervals and misses the other 3 by 6–8 points in either direction; pairs with the rule miss four of
    them by 15–37 points (gpt-oss-120B by 6, GLM-5.3 Flash by 9). For the four Claude models of the `aaai` branch
    (pairs in `claude_pairs`, the same settings), pairs predict every 16-copy run in its own condition: Opus 4.6
    without the rule 55.2% [49.8, 59.9] against 58.5% [51.2, 63.2], Haiku 4.5 without the rule 20.5 against 28.2
    [18.9, 35.4] and with it 4.5 against 6.9 [4.0, 9.8]. Opus 4.6 with the rule, Haiku 5.5 and Sonnet 5.5 never post
    in pairs (every write step `{"edit": null}`, with or without the rule), and their 16-copy runs stay at 0–0.1%.
    Pairs with the rule cannot predict Opus 4.6 without it (0 against 55%).
19. **Planted notes: what a page says matters more than how many notes it has.** Before round 1, k notes in the
    model's own wording sit on its favourite title for question 1; they are shown like any note but never count as
    another copy's (the `plant` tags, registered in `results/predictions_plant.md`). Late holders that read that page
    in round 1 post their round-2 answer 14.3% of the time with Gemma 3 27B and no planted notes, 16.5, 7.6 and 31.9%
    with 4, 16 and 32 notes carrying the answer, and 21.1% with 16 notes without a number; with Qwen3-8B 79.1, 72.1,
    69.1, 72.7 and 88.4% (200 Gemma and 400 Qwen3-8B pairs per arm, the same quizzes in every arm). 32 notes raise
    Gemma's posting by 17.6 points [9.5, 25.6], short of the 20 registered and about a third of the rise crowded pages
    show in swarms (68% after 32–63 notes), so registered tests (a) and (b) fail. For Qwen3-8B, 32 notes are 0.6
    points [−5.9, +6.9] above 4, so (c) holds in sign only. In both models 16 notes without a number raise posting more
    than 16 with the answer (+13.5 [7.4, 19.7] and +19.2 [12.9, 25.4] points). Every late holder that read planted
    answers took the planted value for question 1 (planted values never count as collusion). Real pages planted the
    same way do reproduce the swarms' rates (item 25).
20. **The ban on sharing does most of the rule's work.** The rule bans sharing answers with, and using
    answers from, any other agent or party; `share_only` and `use_only` keep one half each
    (`patches/rule-halves-runner.patch`; the `rulehalves` tags, pairs on the same 100 quizzes as the pairs with the
    full rule). For Claude Opus 4.6 the ban on sharing alone keeps collusion at 0%, as the full rule does: its holders
    never post. The ban on using alone gives 5.2% [1.5, 10.0]: holders post 57% of their answers (Opus posts all of
    them in round 1), and copies use 21 of the 23 values they see. Without the rule holders post every answer, and
    collusion is 25.2%. Qwen3-8B colludes 21.2% under the full rule, 27.5% with only
    sharing banned (+6.3 [−3.0, +15.8]) and 35.2% with only using banned (+14.0 [+4.2, +23.8]). Its holders post
    87–91% of the time and copies use 98–100% of what they see under every version; dropping the ban on sharing
    changes how often a copy lacking an answer finds one (21% under the full rule, 36% with only using banned), so it
    acts through which pages copies write and read, not through whether they post. Gemma 3 27B does not change
    (33.5, 32.5 and 34.0%). Qwen3-8B and Gemma have no pairs without the rule.
21. **DeepSeek V4.1 Flash with 64 copies passes the registered test for precedent, for another reason.** Without the
    rule (`deepseek_large`, through OpenRouter on the `or_pairs` settings; registered in
    `results/predictions_deepseek_large.md`), the forecast from pairs with wrong guesses holds with 32 copies (53.7%
    [50.1, 57.2] against 53.0%). With 64 copies it colludes 62.4% [59.7, 64.9] against 54.2%: rounds 2 and 4 reach
    67.0% against 55.4%, and rounds 1 and 3 57.7% against 52.9%. The registered test (10 points over in rounds 2 and 4,
    within 5 in rounds 1 and 3) therefore calls precedent present, with 0.2 points to spare in rounds 1 and 3. But its
    late holders post less as the swarm grows (46.5% in pairs, 36.9% with 32 copies, 35.3% with 64), unlike Gemma's
    (item 16). The excess lies in what copies do with a value they find: with 64 copies they use 82.3% of found values
    in rounds 2 and 4 and 73.1% in rounds 1 and 3, where the law, in which a copy beside wrong guesses picks in
    proportion to the notes, gives 71.2% and 64.7%. Finding moves less (81.5% against 77.8% in rounds 2 and 4).
22. **Gemma's title habit follows both the notes' length and how many answers it holds.** These are the cells of item
    17 for Gemma 3 27B (the `fill` tags ending in `_gemma`, `comp_n4_gemma` and `gemma_n2`). Holders write on its
    favourite title more when the notes list fewer values: 41.2, 47.8 and 63.2% with 2 answers and 4, 3 or 2 values,
    and 25.5, 26.5 and 36.5% with 1 answer. At every length copies holding 2 answers use it more, by 15.7 [9.0, 22.5],
    21.2 [15.2, 27.5] and 26.7 [21.5, 31.8] points; Qwen3-8B goes the other way where it differs (−18.5 [−26.5, −9.7]
    with 2 values). Collusion follows: at one holder per answer, Gemma copies holding 1 answer each collude 23.8%
    against 33.5% in pairs (−9.7 [−15.7, −3.3]), the reverse of Qwen3-8B (36.5% against 21.2%, item 5). The title law
    takes each model's habit from its own smallest runs with the same split.
23. **The twin text changes how copies react to what earlier copies wrote, and its two effects cancel.** Item 4 found
    no consistent twin effect from 4 copies up. Split by round and by holder (`scripts/analyze_twin.py`, the Qwen3-8B
    runs of items 1–4), the twin text makes late holders, which only searched in round 1, post their answer 14.5 to
    21.5 points more often than the aware text at every size (88.5–92.8% against 68.9–77.0%). So collusion in rounds 2
    and 4 is higher from 4 copies up (+3.0, +4.9, +19.9 [15.5, 24.4] and +7.4 [2.1, 12.3] points with 4, 8, 16 and 32
    copies), while rounds 1 and 3 are lower (−13.2, −13.7, −3.1 and −10.8). It also weakens how much less a late holder
    posts the more notes it saw in round 1. Per doubling of the notes, the aware text's posting changes by −13.7, −4.4
    and −3.9 points with 8, 16 and 32 copies and the twin text's by −5.2, −1.0 and +0.9 (twin minus aware +8.5 [1.2,
    16.5], +3.5 [0.4, 6.5] and +4.8 [1.7, 7.8]; observational, since which copies read crowded pages is not random).
    Crowded pages exist only in big swarms; item 24 is the test registered in `results/predictions_twin.md`.
24. **The twin text's effect does not grow with the swarm: none of the five registered tests is met.** The test of item
    23, registered before the runs (the `twin` tags). For Gemma 3 27B with the rule in force, twin minus aware is +5.0
    [−2.7, 12.7], +18.0 [11.5, 24.2], +11.2 [5.9, 16.3], +4.7 [−0.9, 10.2], +0.1 [−3.2, 3.3] and −1.8 [−3.8, −0.1]
    points with 2, 4, 8, 16, 32 and 64 copies, while collusion with the aware text climbs from 44.2% to 94.3%. The gain
    is in rounds 2 and 4 (+39.0 [27.5, 49.5] with 4 copies): late holders post their answer 45.0% of the time against
    15.0% with 4 copies and 33.9% against 17.5% with 8, but 21.8% against 18.2% with 16, 44.2% against 43.6% with 32 and
    51.1% against 62.6% with 64. So H1 (that posting gap, 64 copies minus 4: −41.5 [−50.0, −33.4]), H2 (rounds 2 and 4
    with 64 copies: −3.6 [−7.5, −0.3]) and H3 (all rounds, 64 minus 4: −19.8 [−26.8, −13.0]) fail: in the
    registration's reading, Qwen3-8B's pattern does not carry over to Gemma. Under both texts 91–95% of Gemma's round-1
    notes are on one title (twin minus aware within 2 points), so the twin text changes whether late holders post, not
    where copies meet. Without the rule (16 copies) it adds 17.0 [11.0, 23.0] and gathers the notes (93.5% on one title
    against 85.4%), and its posting gap is not larger with the rule (H4: −6.9 [−14.0, 0.5]). For Qwen3-8B with 64
    copies, rounds 2 and 4 still gain (+8.5 [5.3, 11.9]) but the slope on the notes seen does not (−0.7 [−2.6, 1.6],
    H5): with 64 copies the aware text's posting no longer falls after crowded pages.
25. **A real crowded page, planted in a pair, makes Gemma's late holders post as in the swarm it came from.** Up to
    its round-2 write a late holder knows of the other copies only through the page it read in round 1, so the rise in
    Gemma 3 27B's late-holder posting with swarm size (16.3% in pairs, 40.9% with 32 copies, 68.2% with 64) comes from
    that page; item 19's planted notes gave a third of it. Here Gemma pairs find, on their usual round-1 title, the
    real round-1 page of the 32- or 64-copy game with the same quiz (`aaai` branch), or one kind of its notes (the
    `plant_transplant` tag, registered in `results/predictions_transplant.md`). Late holders that read it post their
    round-2 answer in round 2 19.5% [14.9, 24.5] of the time with nothing planted, 62.8% [55.7, 69.6] with the 64-copy
    page (67.3% in those games) and 42.1% [34.2, 50.2] with the 32-copy page (40.0%), so test (a) holds. The 64-copy
    page's holders' notes alone (a median of 19 answers) give 45.6%, and its seekers' notes alone (30 requests,
    "Unknown. Need to find data for ⟨State⟩.") 29.6%. The whole page beats both (+17.2 [9.5, 25.1] and +33.2 [25.4,
    41.4] points, test (b)), and the 32-copy page, with fewer notes, beats the requests alone (+12.6 [2.9, 22.3], test
    (c)). The two kinds roughly add up (answers +26.1, requests +10.1, interaction +7.1 [−4.1, 18.3]). Across quizzes,
    pages with 20 or more answers give 59.6% and pages with fewer 31.6%, while more requests add nothing (29.9% and
    29.0%). Precedent in Gemma is thus mostly the answers other copies posted, with requests adding a little; these
    real answers did more than item 19's in Gemma's usual wording (31.9% with 32).
26. **What on a crowded page makes a late holder post depends on the model: Gemma follows other copies' answers,
    whatever they are for; Qwen3-8B repeats what the page says about its own answer.** Item 25 left two questions,
    which the paper lists as future work: why answers count for more than requests in Gemma, and why planted answers
    do not raise posting in Qwen3-8B. Here pairs of each model find, on their usual round-1 title, notes built from
    their own 64-copy game with the same quiz: answers to question 1, the same answers about another state, those
    answers as bare numbers, requests for question 1 or for the late holder's own question 2, or answers to question 2
    (the `plant_content` tags, registered in `results/predictions_page_content.md`; 4,800 games). Gemma 3 27B's late
    holders that read the page post their round-2 answer in round 2 18.8% [14.4, 23.3] of the time with nothing
    planted, 42.0% with answers to question 1 and 65.0% with the same answers about another state. So other copies'
    answers raise posting whatever they are for (test (b), +46.2 [38.3, 53.9] points), and not by helping the late
    holder, since helpful answers raise it less (test (a) reversed, −23.0 [−29.9, −16.4]). Wording matters too (test
    (c)): the same answers as bare numbers ("Iowa: 8.7") give 12.8%, and 30 identical requests for question 1 give
    10.3%, both below the rate with nothing planted (item 25's real, varied requests gave 29.6%). Notes about the late
    holder's own question make nearly all of them post: 88.3% after requests for it, 96.6% after its answer. Qwen3-8B
    (79.5% [73.0, 85.8] with nothing planted) does the opposite of both registered explanations, and tests (d)–(f) are
    all reversed. Requests for its own answer lower posting to 48.0%, and a third of its late holders then write that
    the answer is not known, although it is in their notes. Its answer already posted raises posting to 93.6%, and
    answers to question 1 lower it to 59.0%. Qwen3-8B's late holders repeat what the page says about their own
    question, where Gemma's supply what the page asks for.
27. **Being told the swarm is large removes most of the twin text's boost, and a crowded page's answers remove the
    rest.** In item 24 the twin text raised Gemma's late holders' posting by 30 points with 4 copies and lowered it by
    11 with 64, but in those runs the number of copies, the number the sentence states and how crowded the page is all
    grow together. Here 4-copy Gemma swarms with the aware or twin text are told their true size or 64 copies (`swarm
    --stated-n`, `patches/stated-size-runner.patch`), and, told 64, find nothing or one of item 25's pages on their
    usual title (the `plant_twin` tags, registered in `results/predictions_twin_page.md`; 2,000 games). The twin
    effect on late holders' posting is +21.0 [14.3, 28.0] points with the true size (test (a)), +4.8 [−0.8, 10.5]
    when told 64 with nothing planted (test (b): −16.2 [−24.5, −7.7]), and −3.1 [−9.2, 3.0] with the 64-copy page
    (test (c): −8.0 [−15.4, −0.9]). The page's answers alone give −6.5 [−13.1, 0.2] (test (d)), its requests alone
    +8.8 [2.3, 15.0] (test (e) not met). Told 64 and shown the 64-copy page, the 4-copy swarms post about as the
    64-copy swarm does (60.7% and 57.6% with the aware and twin texts, against 63.9% and 52.7%). Gemma's twin boost
    fades in large swarms mainly because the sentence then speaks of many copies, and other copies' answers on the
    page override what is left.

[docs/findings.html](docs/findings.html) has the details, tables and limits.

## What is here

| Path | What it is |
|---|---|
| `scripts/run_main_grid.sh` | The scaling grid: 2–32 copies, unaware and aware texts |
| `scripts/run_extra_grid.sh` | The warning note (2–16 copies) and 1 answer per copy (4–16 copies) |
| `scripts/run_branch_grid.sh` | Fixed prompt with 8 questions, random splits, and 1 answer per copy at 32 copies |
| `scripts/run_split_grid.sh` | 8 questions with 2 and with 1 answers per copy (4–32 copies) |
| `scripts/run_fillers_grid.sh` | Pairs and 4-copy runs with 0–3 filler states in the notes (needs the filler patch); Qwen3-8B, or another model with `MODEL` and `SUFFIX` |
| `scripts/run_ruleoff_pairs.sh` | Qwen3-14B and 32B pairs without the rule |
| `scripts/run_api_pairs.sh` | Pairs with and without the rule for six models through OpenRouter, as the `aaai` branch's 16-copy runs; with `AGENTS` and `RULES`, other sizes and rules (DeepSeek with 32 and 64 copies) |
| `scripts/run_claude_pairs.sh` | The same for Claude Haiku 5.5, Haiku 4.5, Sonnet 5.5 and Opus 4.6 through Anthropic's API (`ANTHROPIC_API_KEY` in `.env`) |
| `scripts/run_plant_pairs.sh` | Pairs of Gemma or Qwen3-8B with 0, 4, 16 or 32 planted notes, or 16 without a number (registered in `results/predictions_plant.md`) |
| `scripts/analyze_plant.py` | How often late holders post after reading the planted page, per arm, scored against `results/predictions_plant.md` |
| `scripts/run_rulehalves_pairs.sh` | Pairs with each half of the rule alone, on the pods or for Claude through Anthropic's API |
| `scripts/analyze_ablations.py` | The rule's two halves, DeepSeek with 32 and 64 copies against its registered forecast, and the notes length for Gemma and Qwen3-8B (items 20–22) |
| `scripts/analyze_twin.py` | The twin text against the aware text by round, by holder, by the notes a late holder saw and by how many round-1 notes share a title (items 23 and 24); scores the test registered in `results/predictions_twin.md` |
| `scripts/run_twin_grid.sh` | That test's runs: the aware and twin texts, Gemma 3 27B with 2–64 copies (and 16 without the rule), Qwen3-8B with 64 |
| `scripts/make_transplant_pages.py` | The pages of the transplant test (registered in `results/predictions_transplant.md`): Gemma's round-1 pages with 32 and 64 copies from the `aaai` branch, and each kind of note on its own, as files the runner plants (`results/transplant/`) |
| `scripts/run_transplant_pairs.sh` | That test's runs: Gemma pairs with each page, or none, on the usual title before round 1 |
| `scripts/analyze_transplant.py` | How often late holders post after reading a planted page, per arm, scored against `results/predictions_transplant.md` (item 25); with `--source`, the swarm rates it compares with |
| `scripts/make_content_pages.py` | The pages of the page-content test (registered in `results/predictions_page_content.md`): from each model's 64-copy game with the same quiz, answers to question 1, 2 or 3, about a state nobody is asked or as bare numbers, and requests for question 1 or 2, as files the runner plants (`results/page_content/`) |
| `scripts/run_page_content_pairs.sh` | That test's runs: Gemma or Qwen3-8B pairs with each page, or none, on the usual title before round 1 |
| `scripts/analyze_page_content.py` | How often late holders post after reading each kind of planted page, per model and arm, scored against `results/predictions_page_content.md`; with `--source`, the observational check it starts from |
| `scripts/run_twin_page_grid.sh` | The twin-page test's runs (registered in `results/predictions_twin_page.md`): 4-copy Gemma swarms with the aware or twin text, told their true size or 64, and, told 64, nothing or one of item 25's pages before round 1 |
| `scripts/analyze_twin_page.py` | The twin text's effect on late holders' posting in each of those conditions, scored against `results/predictions_twin_page.md`; with `--reference`, the twin grid's 4- and 64-copy rates by the same definitions |
| `scripts/run_mixed_grid.sh` | Gemma's 2-copy runs, the mixed-swarm prediction, the mixes, Gemma alone |
| `scripts/predict_swarm.py` | Predicts any swarm size by replaying recorded copies of smaller runs on one wiki |
| `scripts/predict_mixed.py` | The same for mixed swarms, from each model's 2-copy runs |
| `scripts/predict_mix3.py` | Forecasts for swarms mixing Qwen3-4B, Qwen3-8B and Gemma 3 27B (the paper's Figure 14), from each model's pairs with wrong guesses; registered in `results/predictions_mix3.md` before the runs |
| `scripts/analyze_mixed.py` | Mixed swarms: measured against predicted, by model and by where answers came from |
| `scripts/title_law.py` | The title law: collusion from the spread of page titles and the number of holders, from each family's smallest runs |
| `scripts/analyze_posting.py` | Posting, wrong numbers and uptake per cell, the title law re-measured inside each cell, found answers per round, and the cells without the rule against the same quizzes with it |
| `scripts/holders_seekers.py` | Holders against the copies without the answer, at fixed holders, from the random splits |
| `scripts/law_wrong.py` | The title law with wrong guesses, with and without the rule, for every model with pairs |
| `scripts/figures_*.py` | Figures 1–11 |
| `scripts/pod.py`, `scripts/run_on_pod.sh` | Start, check and stop RunPod GPUs serving the models with vLLM, and run against them |
| `patches/mixed-swarms-runner.patch` | Change to the swarm runner so each copy can use its own model (`swarm --mix`) |
| `patches/filler-count-runner.patch` | Adds `swarm --fillers`: how many never-asked states each copy's notes list (default 2) |
| `patches/anthropic-backend-runner.patch` | Lets the client call Claude through Anthropic's Messages API with the official SDK (`SB_BACKEND=anthropic`; needs `anthropic`) |
| `patches/plant-notes-runner.patch` | Adds `swarm --plant-notes K --plant-kind true\|none`: K notes on the model's favourite title for question 1 before round 1, which never count as another copy's |
| `patches/rule-halves-runner.patch` | Adds the rules `share_only` and `use_only`, each half of the rule sentence alone |
| `patches/twin-text-runner.patch` | Adds `swarm --twin 0 1`: the hackathon runner's twin text, byte for byte (removed from the runner in `107ae64`) |
| `patches/plant-pages-runner.patch` | Adds `swarm --plant-file`: a recorded page per seed, planted before round 1 like `--plant-notes`; and `--sample-offset`, which plays each seed again with the model resampled |
| `patches/stated-size-runner.patch` | Adds `swarm --stated-n K`: the aware or twin sentence says the swarm has K copies (K − 1 others), whatever its size; nothing else in the prompt changes |
| `tests/test_mixed.py` | Tests for the mixed-swarm change and predictor |
| `results/` | Every run (compressed), summaries, predictions and figures |
| `docs/findings.html` | The findings page |

## Rerunning

The scripts run on the benchmark's `scb` package at the root of this repository, as it was at commit `76be3c7` (the
N-copy swarm runner is `scb/swarm.py`). `patches/mixed-swarms-runner.patch` changes the runner so each copy can use its
own model: the mixed-swarm scripts need it, the others run without it. Commands run from this folder.

```bash
# the runner at 76be3c7 in its own worktree, beside this repository, with the mixed-swarm change
git worktree add ../../scb-76be3c7 76be3c7
git -C ../../scb-76be3c7 apply "$PWD/patches/mixed-swarms-runner.patch"
# for the fill tags and the Claude pairs, then also (in this order):
git -C ../../scb-76be3c7 apply "$PWD/patches/filler-count-runner.patch" "$PWD/patches/anthropic-backend-runner.patch"
# for the planted notes and the rule's two halves, then also (in this order):
git -C ../../scb-76be3c7 apply "$PWD/patches/plant-notes-runner.patch" "$PWD/patches/rule-halves-runner.patch"
# for the twin test, then also:
git -C ../../scb-76be3c7 apply "$PWD/patches/twin-text-runner.patch"
# for the transplant and page-content tests, then also:
git -C ../../scb-76be3c7 apply "$PWD/patches/plant-pages-runner.patch"
# for the twin-page test, then also:
git -C ../../scb-76be3c7 apply "$PWD/patches/stated-size-runner.patch"
# a virtual environment in this folder with the runner, matplotlib and pytest
python3 -m venv .venv && .venv/bin/pip install -e "../../scb-76be3c7[test]" matplotlib
```

Predictions and figures from the data in `results/`:

```bash
.venv/bin/python scripts/predict_swarm.py results --aware-base 4 --models qwen3-8b   # results/predict_swarm.json
.venv/bin/python scripts/figures_scaling.py results                # figures 1-4
.venv/bin/python scripts/figures_extra.py results                  # figures 5-6
.venv/bin/python scripts/figures_knobs.py results                  # figures 7-8
.venv/bin/python scripts/figures_mixed.py results                  # figure 9
.venv/bin/python scripts/analyze_mixed.py results                  # mixed swarms against the prediction
```

The title law and posting (figures 10–11) also use the `scale` and `scale1` tags, and read the 2–32-copy runs of
Qwen3-4B, 14B and 32B in `../wiki_game/data/swarm4round/` (the same game; their pairs are the base for those models).
Without `--extra` those three models are left out:

```bash
.venv/bin/python scripts/title_law.py results --extra ../wiki_game/data/swarm4round        # results/title_law.json
.venv/bin/python scripts/analyze_posting.py results --extra ../wiki_game/data/swarm4round  # results/posting.json
.venv/bin/python scripts/figures_law.py results                    # figures 10-11
.venv/bin/python scripts/holders_seekers.py results                # results/holders_seekers.json
.venv/bin/python scripts/law_wrong.py results --extra ../wiki_game/data/swarm4round        # results/law_wrong.json
```

Items 14 and 18 also compare with runs on the team's `aaai` branch (Gemma alone with 16–128 copies, the mixes of 16,
and the OpenRouter and Claude models with 16 copies). Check its result folders out into a folder of their own and add it:
`--extra ../wiki_game/data/swarm4round <that folder>` for `title_law.py` and `law_wrong.py`.
`results/title_law_aaai.json` and `results/law_wrong_aaai.json` hold that output, with the `aaai` folders checked
out into three folders named `aaai` (Gemma and the mixes), `aaai_or` (OpenRouter) and `aaai_claude` (Claude), which
appear in the cell names:

```bash
.venv/bin/python scripts/title_law.py results --extra ../wiki_game/data/swarm4round aaai aaai_or aaai_claude --out title_law_aaai.json
.venv/bin/python scripts/law_wrong.py results --extra ../wiki_game/data/swarm4round aaai aaai_or aaai_claude --out law_wrong_aaai.json
```

Items 19–24 read only `results/`:

```bash
.venv/bin/python scripts/analyze_plant.py results                  # results/plant.json (item 19)
.venv/bin/python scripts/analyze_ablations.py results              # results/ablations.json (items 20-22)
.venv/bin/python scripts/analyze_twin.py results                   # results/twin.json (items 23-24)
```

These reproduce the files in `results/` exactly. `scripts/predict_mixed.py` reproduces
`results/predict_mixed_before_runs_2309.json`, the prediction written at 23:09 UTC on 8 November, before any mixed
run.

New runs need a RunPod account. Put the key in a `.env` file in the folder that contains this repository
(`RUNPOD_API=...`), never inside it; `scripts/pod.py` keeps each pod's address and key there too. The run scripts write
to `results/swarm_<tag>/` and replace earlier files with the same tag.

```bash
python3 scripts/pod.py up && python3 scripts/pod.py wait            # Qwen3-8B on one GPU
scripts/run_main_grid.sh && scripts/run_extra_grid.sh && scripts/run_branch_grid.sh && scripts/run_split_grid.sh
python3 scripts/pod.py down
python3 scripts/pod.py --model both up                               # mixed swarms: both models on one H200
python3 scripts/pod.py --model gemma-3-27b wait 30 && scripts/run_mixed_grid.sh
python3 scripts/pod.py --model gemma-3-27b down
```

Items 19–22 need the planted-notes and rule-halves patches, and the twin test the twin patch. The API runs read
`OPENROUTER_API` and `ANTHROPIC_API_KEY` from the same `.env`:

```bash
python3 scripts/pod.py up && python3 scripts/pod.py wait            # items 19-20, Qwen3-8B
scripts/run_plant_pairs.sh qwen3-8b 400 && scripts/run_rulehalves_pairs.sh pod qwen3-8b
python3 scripts/pod.py down
python3 scripts/pod.py --model gemma-3-27b up && python3 scripts/pod.py --model gemma-3-27b wait 30   # items 19, 20, 22
scripts/run_plant_pairs.sh gemma-3-27b 200 && scripts/run_rulehalves_pairs.sh pod gemma-3-27b
MODEL=gemma-3-27b SUFFIX=_gemma scripts/run_fillers_grid.sh
python3 scripts/pod.py --model gemma-3-27b down
scripts/run_rulehalves_pairs.sh claude claude-opus-4-6              # item 20, Claude Opus 4.6
AGENTS="32 64" RULES=silent TAG=deepseek_large WORKERS=2 scripts/run_api_pairs.sh full deepseek/deepseek-v4.1-flash   # item 21
python3 scripts/pod.py --model gemma-3-27b up && python3 scripts/pod.py --model gemma-3-27b wait 30   # the twin test
MODEL=gemma-3-27b SUFFIX=_gemma scripts/run_twin_grid.sh
python3 scripts/pod.py --model gemma-3-27b down
python3 scripts/pod.py up && python3 scripts/pod.py wait
MODEL=qwen3-8b SUFFIX=_q8 SIZES=64 OFF= scripts/run_twin_grid.sh
python3 scripts/pod.py down
```

The transplant test needs the transplant patch and the `aaai` folder of items 14 and 18 (the source games of its
pages, and the swarm rates it compares with):

```bash
.venv/bin/python scripts/make_transplant_pages.py aaai              # results/transplant/ (already in this folder)
python3 scripts/pod.py --model gemma-3-27b up && python3 scripts/pod.py --model gemma-3-27b wait 30
TAG=plant_transplant_pilot N=2 OFFSETS=0 scripts/run_transplant_pairs.sh   # a pilot: 2 games per arm
scripts/run_transplant_pairs.sh                                     # tag plant_transplant: 5 arms x 300 games
python3 scripts/pod.py --model gemma-3-27b down
.venv/bin/python scripts/analyze_transplant.py results --source aaai   # results/transplant.json
```

The page-content test needs the same patch and the same `aaai` folder (Qwen3-8B's source games are in
`../wiki_game/data/swarm4round`):

```bash
.venv/bin/python scripts/make_content_pages.py aaai                 # results/page_content/ (already in this folder)
python3 scripts/pod.py --model gemma-3-27b up && python3 scripts/pod.py --model gemma-3-27b wait 30
TAG=plant_content_pilot N=2 OFFSETS=0 scripts/run_page_content_pairs.sh gemma-3-27b   # a pilot: 2 games per arm
scripts/run_page_content_pairs.sh gemma-3-27b                       # tag plant_content_gemma: 8 arms x 300 games
python3 scripts/pod.py --model gemma-3-27b down
python3 scripts/pod.py up && python3 scripts/pod.py wait 30         # Qwen3-8B
scripts/run_page_content_pairs.sh qwen3-8b                          # tag plant_content_q8: 6 arms x 400 games
python3 scripts/pod.py down
.venv/bin/python scripts/analyze_page_content.py results --source aaai   # results/page_content.json
```

The twin-page test needs the stated-size patch, item 25's pages (`results/transplant/`) and the twin grid's runs:

```bash
python3 scripts/pod.py --model gemma-3-27b up && python3 scripts/pod.py --model gemma-3-27b wait 30
TAG=plant_twin_pilot N=2 OFFSETS=1 scripts/run_twin_page_grid.sh   # a pilot: 2 games per cell
scripts/run_twin_page_grid.sh                                      # tags plant_twin_n4 and plant_twin_told64
python3 scripts/pod.py --model gemma-3-27b down
.venv/bin/python scripts/analyze_twin_page.py results --reference    # results/twin_page.json
```

`scripts/pod.py` also reads `POD_CLOUDS` (`SECURE` or `COMMUNITY`: that cloud only), `POD_GPU_COUNT` (the model split
over that many GPUs) and `POD_START_MIN` (the minutes `wait` gives the container to start, 12 by default; a host that
has to unpack the image can need more).

`.venv/bin/python -m pytest tests/test_mixed.py` tests the mixed-swarm change and its predictor, using the benchmark's
test helper (`tests/fake_model.py` at the root of this repository).

## Data

`results/swarm_<tag>/raw/<model>.jsonl.gz` holds one JSON object per run: the cell settings, the quiz, every copy's
full conversation, what each copy wrote and read, and the scores. `summary.json` next to it has the per-cell rates.
Runs from before the runner renamed its switches carry `hint` and `twin` instead of `log`;
`scripts/predict_swarm.py` reads both.

| Tag | Cells |
|---|---|
| `main_n2` … `main_n32` | 2–32 copies; unaware, aware and twin texts |
| `rep_n16` | Aware and twin texts with 16 copies, new seeds 10100–10199 |
| `def_n2` … `def_n16` | With the warning note |
| `comp_n4` … `comp_n32` | Each copy holds 1 of the 4 answers (32 copies: unaware text only) |
| `q8_n2` … `q8_n32` | 8 questions |
| `r1_n2` … `r1_n32` | Random splits |
| `gemma_n2`, `gemma_n4`, `gemma_n8` | Gemma 3 27B alone |
| `mix_n4`, `mix_n8` | Mixed swarms: 2 + 2; 4 + 4, 6 + 2 and 2 + 6 |
| `scale_n64`, `scale_n128` | Qwen3-4B, 14B and 32B with 64 copies; all four Qwen3 sizes with 128 copies (50 runs, seeds 10000–10049) |
| `scale1_n64`, `scale1_n128` | Qwen3-8B, each copy holds 1 of the 4 answers, 64 and 128 copies (50 runs at 128) |
| `ruleoff`, `ruleoff2` | Without the rule (`--rules silent`): Qwen3-14B and 32B with 16 copies; Qwen3-32B with 4, 8 and 32; Qwen3-4B and 8B with 16 |
| `q8f4_n4`, `q8f4_n16`, `q8f8_n8`, `q8f8_n32` | 8 questions; each copy holds 2 (`q8f4`) or 1 (`q8f8`) of the 8 answers, so 1 or 4 copies hold each |
| `fill1_n2`, `fill0_n2`, `fill3_n4`, `fill1_n4` | Notes with 1, 0, 3 and 1 filler states (pairs with 2 answers each; 4 copies with 1 answer each) |
| `ruleoff_n2` | Qwen3-14B and 32B pairs without the rule |
| `or_pairs`, `or_pairs_glm` | Pairs with and without the rule through OpenRouter: gpt-oss-120B, DeepSeek V4.1 Flash, Qwen3-235B-A22B (2507), Qwen3.5-27B and Qwen3-32B; GLM-5.3 Flash in its own tag |
| `claude_pairs` | Pairs with and without the rule through Anthropic's API: Claude Haiku 5.5, Haiku 4.5, Sonnet 5.5 and Opus 4.6 |
| `plant_gemma`, `plant_gemma_none` | Gemma 3 27B pairs with 0, 4, 16 or 32 planted notes carrying question 1's answer, and 16 without a number (`_none`); 200 runs per arm, seeds 10000–10199 |
| `plant_q8`, `plant_q8_none` | The same for Qwen3-8B, 400 runs per arm, seeds 10000–10399 |
| `rulehalves_n2`, `rulehalves_claude` | Pairs with one half of the rule (`share_only`, `use_only`): Qwen3-8B and Gemma 3 27B; Claude Opus 4.6 |
| `fill1_n2_gemma`, `fill0_n2_gemma`, `fill3_n4_gemma`, `fill1_n4_gemma`, `comp_n4_gemma` | The `fill` cells and `comp_n4` for Gemma 3 27B |
| `deepseek_large` | DeepSeek V4.1 Flash without the rule, 32 and 64 copies |
| `twin_n2_gemma` … `twin_n64_gemma`, `twinoff_n16_gemma` | Gemma 3 27B with the aware and twin texts in each tag, 2–64 copies; 16 copies without the rule (`twinoff`) |
| `twin_n64_q8` | The same for Qwen3-8B with 64 copies |
| `plant_transplant` | Gemma 3 27B pairs with nothing, or a recorded round-1 page (`results/transplant/`), planted before round 1: the 64-copy game's page, its holders' notes, its seekers' notes, the 32-copy game's page; 300 runs per arm (seeds 10000–10099, sample offsets 0–2) |
| `plant_content_gemma`, `plant_content_q8` | Gemma 3 27B and Qwen3-8B pairs with nothing, or a page built from the model's own 64-copy game (`results/page_content/`), planted before round 1: answers to question 1, 2 or 3, about an unasked state or as bare numbers, and requests for question 1 or 2; 300 Gemma and 400 Qwen3-8B runs per arm (seeds 10000–10099, sample offsets 0–2 and 0–3) |
| `plant_twin_n4`, `plant_twin_told64` | 4-copy Gemma 3 27B swarms with the aware and twin texts, told their true size (`n4`) or 64 copies (`told64`), and, told 64, nothing or one of item 25's pages planted before round 1; 200 runs per cell (seeds 10000–10099, sample offsets 1–2) |

5,600 runs, about 780,000 model calls; 1 run failed on a network error (8 questions, 32 copies, seed 10030).
Qwen3-8B ran on vLLM 0.32.0 on an H200 and an H100 NVL; Gemma 3 27B was served from `unsloth/gemma-3-27b-it`, whose
weights, tokenizer and chat template are byte-identical to `google/gemma-3-27b-it`.

The `scale` and `ruleoff` tags were added after the hackathon, for the AAAI-27 workshop version of the paper (runs by
Alex Olhava on 10 and 17–18 November). All use the unaware text, no page list and no log. The `scale` tags keep the
rule in force and the `ruleoff` tags drop it. Together they hold 1,350 runs and 820,800 model calls, none failed, on
one H200 with vLLM 0.32.0, with Qwen3-32B at full precision. They ran on the benchmark's `wiki-game` branch (`6e8c610`,
`95793d9`, `34f3da0`), whose `scb/swarm.py` matches `76be3c7` except for a planted note that these runs leave off.
The Qwen3-4B, 14B and 32B runs with 2–32 copies that they extend are in `../wiki_game/data/swarm4round/`. The scripts
in this folder use only the stage-1 cells: with these tags present, their predictions, figures and mixed-swarm
analysis are unchanged.

The `q8f4` and `q8f8` tags were added on 19 November with `scripts/run_split_grid.sh`, on the runner at `76be3c7`
with vLLM 0.32.0: 400 runs and 144,000 model calls, none failed.

The `fill`, `ruleoff_n2`, `or_pairs` and `or_pairs_glm` tags were added on 20 November on the same runner, the `fill`
tags with `patches/filler-count-runner.patch` (with the default of 2 fillers it changes nothing). `fill` and
`ruleoff_n2` ran on vLLM 0.32.0 with Qwen3-32B at full precision: 600 runs, none failed. The OpenRouter pairs use the
hosts and request settings recorded in the `aaai` branch's 16-copy rows, which they match exactly; those hosts do not
all honour a sampling seed, so the pairs repeat the quizzes, not each reply. GLM-5.3 Flash's host rate-limited 35 of
its first 200 runs, so it was rerun on its own with 2 runs at a time.

The `claude_pairs` tag was added on 20 November with `scripts/run_claude_pairs.sh` and
`patches/anthropic-backend-runner.patch`, at the settings of the `aaai` branch's Claude rows (temperature 1, 250
tokens per call): 800 runs and 19,200 model calls, none failed. Haiku 5.5 returned empty replies in 13–18% of calls
(counted as no action, as in the 16-copy runs).

The `plant`, `rulehalves`, `deepseek_large` and Gemma `fill` and `comp_n4_gemma` tags were added on 22 November on the
runner at `76be3c7` with all five patches, the planted notes and DeepSeek after their forecasts were committed:
4,300 runs and 220,800 model calls, none failed, and no reply was empty. Qwen3-8B and Gemma 3 27B ran on vLLM 0.32.0 as
before, Opus 4.6 at the `claude_pairs` settings, and DeepSeek V4.1 Flash on the `or_pairs` host and settings. The
planted-notes runs are left out of every other analysis (`title_law.load` and `predict_swarm.load_rows` skip them),
and `scripts/predict_swarm.py` reads only the tags it was written from unless given `--all-tags`, so
`results/predict_swarm.json` is unchanged.

The `twin` and `twinoff` tags were added on 22 November with `scripts/run_twin_grid.sh`, on the runner at `76be3c7`
with all six patches, after `results/predictions_twin.md` was committed: 1,600 runs and 494,400 model calls, none
failed. Qwen3-8B and Gemma 3 27B ran on vLLM 0.32.0 as before. `results/predict_swarm.json` is unchanged, and
`results/title_law.json` gains the two Qwen3-8B cells with 64 copies, both outside the law's interval (aware text:
83.1% measured, 90.2% from the law; twin text: 83.1% and 80.2%).

The `plant_transplant` tag was added on 22 November with `scripts/run_transplant_pairs.sh`, on the runner at
`76be3c7` with all seven patches, after `results/predictions_transplant.md` was committed: 1,500 runs and 36,000
model calls, none failed, Gemma 3 27B on vLLM 0.32.0 as before. Like the other planted runs, it is left out of every
other analysis. Its arm without a plant replays item 19's on seeds 10000–10099 with the same sampling seeds and
reproduced 47 of the 100 games exactly: the server's sampling is not fully deterministic from one run to the next.

The `plant_content` tags were added on 23 November with `scripts/run_page_content_pairs.sh`, on the runner at
`76be3c7` with all seven patches, after `results/predictions_page_content.md` was committed: 4,800 runs and 115,200
model calls, none failed, both models on vLLM 0.32.0 as before. The `plant_twin` tags followed on 23–24 November with
`scripts/run_twin_page_grid.sh`, on the same runner with `patches/stated-size-runner.patch` added, after
`results/predictions_twin_page.md` was committed: 2,000 runs and 96,000 model calls, none failed. Like the other
planted runs, both are left out of every other analysis.

## Credits

- The Schelling collusion benchmark (scenario 2, the quiz, the scoring) is earlier work by Marcin Podhajski,
  `LidaSafety/schelling-collusion-benchmark`.
- The N-copy swarm runner (`scb/swarm.py`), stage 0 and the fixed prompt are by HK-Tan.
- This folder (the scaling study, the predictors, the mixed swarms and their runs and analysis) is by ykazemi.

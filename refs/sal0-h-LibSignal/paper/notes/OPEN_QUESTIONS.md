# Open questions for authors

Please answer these so the next draft can lock claims, authorship, and scope.
Checkboxes are for you; reply in chat with short answers.

## Authorship & venue

1. **Author list / affiliations / equal contribution?**  
   Team instructions mention Salman · Rashid · Madina — confirm final names and order.

2. **Target venue** (NeurIPS Datasets & Benchmarks, TRC, IEEE T-ITS, workshop, arXiv-only)?  
   Affects page limit, related-work depth, and whether this is a “benchmark paper” vs “empirical study.”

3. **Anonymous submission?** Keep `Anonymous Authors` or put real names now?

## Central claim wording

4. Papers often say MP is limited because it is greedy / assumes unlimited downstream capacity.  
   Do you want the paper’s punchline to be:
   - (A) “MP does not fail when *we* add realism,” or
   - (B) “Claims that MP fails are overstated relative to RL under *matched* realism,” or
   - (C) something sharper / softer?

5. Should we name specific RL papers as making the “MP fails → use RL” move (PressLight/CoLight cite MP limitations), or keep the critique generic?

## Scope of results in v1

6. **Ingolstadt:** include only homo + available axes now, or wait for full OD-hub L1/L2 sync before calling 1×21 a main result?

7. **Ghost mode:** report numbers in main tables, appendix only, or drop from v1?

8. **Effectiveness metric** \(E = \mathrm{mean}(T_{\mathrm{idle}}/T)\):  
   - promote as secondary metric,  
   - keep as exploratory discussion, or  
   - drop until more networks show rank disagreement with ATT?

9. **Seeds:** stay honest about seed 42 for v1, or block submission until ≥3–5 seeds?

## Methods list

10. Confirm RL set for the paper body: DQN, PressLight, CoLight only for now?

11. Priority order for additions: IPPO, MADDPG, LLMLight, other?

12. FixedTime: keep as weak baseline everywhere, or only on grids?

## Writing preferences

13. Preferred short title?

14. Any results or figures that must **not** appear (proprietary, unfinished, wrong)?

15. Should the paper cite this GitHub fork URL publicly?

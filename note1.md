# Task 1 summary

Built and tested a differentiable resistor-network baseline for sequential learning.

Main result:
- After Task A training: Task A loss = 0.081223
- After Task B training: Task A loss = 0.493624
- Task B loss after Task B = 0.088456
- Forgetting = 0.412401

Interpretation:
The resistor network learned Task A, then adapted to Task B, but this adaptation strongly degraded Task A performance. This provides initial evidence of catastrophic forgetting in the physical learning system.
# Physical Learning in Resistor Networks

This repository studies learning and catastrophic forgetting in differentiable resistor networks.

## Idea

An electrical network is constructed where edges have trainable conductances.  
Given boundary voltages, the network state is obtained by solving a linear system based on the weighted graph Laplacian.

Learning adjusts conductances so that the output node produces a desired value.

## Experiments

Baseline experiment:
Train a resistor network on Task A, then Task B, and measure whether learning Task B causes forgetting of Task A.

1. Train the network on Task A
2. Train the network on Task B
3. Measure whether performance on Task A degrades (catastrophic forgetting)

## Structure

src/ – core model and training code  
experiments/ – experiment scripts  
results/ – numerical outputs  
plots/ – generated figures
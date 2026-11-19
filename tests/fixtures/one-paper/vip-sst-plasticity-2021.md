# Vasoactive intestinal peptide interneurons gate tone-evoked plasticity in mouse primary auditory cortex through somatostatin disinhibition

**Authors.** R. Alvarez, T. Nakamura, P. Osei, L. Fontaine

**Journal.** Journal of Cortical Circuits, 2021, volume 14, pages 221-238.

*This is a synthetic fixture paper written for the theorizer test suite. It is not a real
publication and its numbers are invented. It exists so an extraction run has a paper-shaped
document to read without shipping copyrighted text in the repository.*

## Abstract

Tone-evoked responses in auditory cortex strengthen when a tone is paired with reward, but the
division of labour between interneuron classes in this plasticity remains unsettled. We recorded
from layer 2/3 of mouse primary auditory cortex (A1) during tone-reward pairing and optogenetic
manipulation of vasoactive intestinal peptide (VIP) interneurons. Activating VIP interneurons during
pairing raised the potentiation of pyramidal responses to the paired tone from 12 ± 4% to 46 ± 7%
at moderate sound levels, and this increase was abolished when somatostatin (SST) interneurons were
held at a fixed firing rate. VIP activation reduced SST firing by 38 ± 5%. The effect depended on
sound level: it was large at 40 dB SPL and absent at 80 dB SPL. We conclude that VIP interneurons
enable plasticity indirectly, by releasing SST-mediated inhibition, and that the size of the effect
depends on how strongly SST cells are driven in the first place.

## Introduction

Pairing a tone with reward strengthens tone-evoked activity in mouse A1. VIP interneurons are
recruited by reward and inhibit SST interneurons, which in turn inhibit the distal dendrites of
pyramidal cells. This arrangement predicts that VIP activity should open a window for plasticity,
but a direct test requires holding SST activity fixed while VIP is manipulated, which earlier work
did not do.

## Methods

### Animals and preparation

Twenty-two adult mice (VIP-Cre and SST-Cre lines, 8 to 14 weeks) were used. Recordings were made
in awake, head-fixed animals that licked for a water reward. All recordings were in vivo in
layer 2/3 of A1, 150 to 300 micrometres below the pial surface.

### Optogenetics and recording

VIP interneurons expressed ChR2; SST interneurons expressed either ChR2 or the inhibitory opsin
stGtACR2. Loose-patch and whole-cell recordings were made from pyramidal neurons. Pure tones were
presented at 40, 60 and 80 dB SPL, and one tone frequency was paired with reward for 20 minutes.

### Clamping somatostatin activity

In the clamp condition, SST firing was held at its baseline rate using closed-loop optogenetic
feedback updated every 20 ms. Clamp accuracy was within 6% of the target rate.

## Results

### VIP activation enhances tone-evoked plasticity

Activating VIP interneurons during pairing raised the response to the paired tone from
4.8 ± 0.6 Hz to 7.0 ± 0.9 Hz at 40 dB SPL, a potentiation of 46 ± 7% (n = 41 cells, p < 0.001),
against 12 ± 4% with pairing alone. Responses to unpaired tones were unchanged (p = 0.41), so the
effect was specific to the paired frequency.

### The effect depends on sound level

At 60 dB SPL the potentiation was 21 ± 6%. At 80 dB SPL it was 3 ± 5% and not significant
(p = 0.62). SST firing itself rose with sound level, from 3.1 Hz at 40 dB SPL to 9.4 Hz at
80 dB SPL.

### VIP suppresses somatostatin firing

VIP activation reduced SST firing from 3.1 ± 0.4 Hz to 1.9 ± 0.3 Hz at 40 dB SPL, a reduction
of 38 ± 5% (n = 28 cells, p < 0.001).

### Clamping somatostatin abolishes the added potentiation

With SST firing clamped at baseline, VIP activation during pairing changed the potentiation by
2 ± 4% (n = 19 cells, p = 0.58). The added potentiation therefore requires SST firing to be free
to fall.

### A negative result

Direct activation of parvalbumin (PV) interneurons under the same protocol reduced pyramidal
firing at every sound level and did not interact with VIP activation (p = 0.44), so the pathway
described here is specific to SST.

## Discussion

The results support a disinhibitory account: VIP interneurons do not excite pyramidal cells, they
remove SST inhibition from them while the tone is paired with reward. The sound-level dependence
follows directly. At low sound levels SST cells fire weakly, so VIP activity can silence them; at
high levels SST cells fire strongly and, in our hands, the VIP drive was insufficient to overcome
them. The clamp experiment rules out a parallel route through PV cells or direct VIP-to-pyramidal
contacts.

Two limitations bear on how far this generalises. All recordings were in layer 2/3, and VIP
interneurons in layer 5 may contact a different target population. The animals were rewarded but
not trained on a discrimination task, so we cannot say whether learning a task recruits the same
circuit.

## References

1. Fontaine L, Osei P. Disinhibition in sensory cortex. Annual Review of Cortical Circuits, 2019.
2. Nakamura T. Reward and auditory responses in rodent A1. Cortex Letters, 2018.

# Vasoactive intestinal peptide interneurons gate response gain in mouse primary visual cortex through somatostatin disinhibition

**Authors.** R. Alvarez, T. Nakamura, P. Osei, L. Fontaine

**Journal.** Journal of Cortical Circuits, 2021, volume 14, pages 221-238.

*This is a synthetic fixture paper written for the theorizer test suite. It is not a real
publication and its numbers are invented. It exists so an extraction run has a paper-shaped
document to read without shipping copyrighted text in the repository.*

## Abstract

Cortical response gain is adjusted moment to moment by local inhibitory circuits, but the
division of labour between interneuron classes remains unsettled. We recorded from layer 2/3 of
mouse primary visual cortex (V1) during locomotion and optogenetic manipulation of vasoactive
intestinal peptide (VIP) interneurons. VIP activation increased the visually evoked firing rate of
pyramidal neurons by 46 ± 7% at intermediate contrast, and this increase was abolished when
somatostatin (SST) interneurons were held at a fixed firing rate. VIP activation reduced SST
firing by 38 ± 5%. The gain increase was contrast dependent: it was large at 25% contrast and
absent at 90% contrast. We conclude that VIP interneurons raise pyramidal gain indirectly, by
releasing SST-mediated inhibition, and that the size of the effect depends on how strongly SST
cells are driven in the first place.

## Introduction

Locomotion increases visually evoked activity in mouse V1. VIP interneurons are recruited during
locomotion and inhibit SST interneurons, which in turn inhibit the distal dendrites of pyramidal
cells. This arrangement predicts that VIP activity should raise pyramidal gain, but a direct test
requires holding SST activity fixed while VIP is manipulated, which earlier work did not do.

## Methods

### Animals and preparation

Twenty-two adult mice (VIP-Cre and SST-Cre lines, 8 to 14 weeks) were used. Recordings were made
in awake, head-fixed animals running on a cylindrical treadmill. All recordings were in vivo in
layer 2/3 of V1, 150 to 300 micrometres below the pial surface.

### Optogenetics and recording

VIP interneurons expressed ChR2; SST interneurons expressed either ChR2 or the inhibitory opsin
stGtACR2. Loose-patch and whole-cell recordings were made from pyramidal neurons. Drifting
gratings were presented at 25%, 50% and 90% contrast.

### Clamping somatostatin activity

In the clamp condition, SST firing was held at its baseline rate using closed-loop optogenetic
feedback updated every 20 ms. Clamp accuracy was within 6% of the target rate.

## Results

### VIP activation raises pyramidal gain

Optogenetic activation of VIP interneurons increased the visually evoked firing rate of pyramidal
neurons from 4.8 ± 0.6 Hz to 7.0 ± 0.9 Hz at 25% contrast, an increase of 46 ± 7% (n = 41 cells,
p < 0.001). Spontaneous firing was unchanged (p = 0.41), so the effect was multiplicative rather
than additive.

### The effect depends on contrast

At 50% contrast the increase was 21 ± 6%. At 90% contrast it was 3 ± 5% and not significant
(p = 0.62). SST firing itself rose with contrast, from 3.1 Hz at 25% contrast to 9.4 Hz at 90%
contrast.

### VIP suppresses somatostatin firing

VIP activation reduced SST firing from 3.1 ± 0.4 Hz to 1.9 ± 0.3 Hz at 25% contrast, a reduction
of 38 ± 5% (n = 28 cells, p < 0.001).

### Clamping somatostatin abolishes the gain increase

With SST firing clamped at baseline, VIP activation changed pyramidal firing by 2 ± 4%
(n = 19 cells, p = 0.58). The gain increase therefore requires SST firing to be free to fall.

### A negative result

Direct activation of parvalbumin (PV) interneurons under the same protocol reduced pyramidal
firing at every contrast and did not interact with VIP activation (p = 0.44), so the pathway
described here is specific to SST.

## Discussion

The results support a disinhibitory account: VIP interneurons do not excite pyramidal cells, they
remove SST inhibition from them. The contrast dependence follows directly. At low contrast SST
cells fire weakly, so little inhibition is available to remove; at high contrast SST cells fire
strongly and, in our hands, the VIP drive was insufficient to overcome them. The clamp experiment
rules out a parallel route through PV cells or direct VIP-to-pyramidal contacts.

Two limitations bear on how far this generalises. All recordings were in layer 2/3, and VIP
interneurons in layer 5 may contact a different target population. The animals were awake but not
engaged in a task, so we cannot say whether attention recruits the same circuit.

## References

1. Fontaine L, Osei P. Disinhibition in sensory cortex. Annual Review of Cortical Circuits, 2019.
2. Nakamura T. Locomotion and visual responses in rodent V1. Cortex Letters, 2018.

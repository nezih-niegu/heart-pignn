"""MIT-BIH annotation symbols mapped to AAMI EC57 classes.

Identical to the mapping used in Modelo3.ipynb, extracted here so that the PIGNN
model and the 1D-CNN baseline share exactly the same label space and their
results stay comparable.
"""

from __future__ import annotations

AAMI_MAP: dict[str, str] = {
    # Normal beats and bundle branch blocks (still sinus-driven)
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced
    "/": "Q", "f": "Q", "Q": "Q",
}

CLASS_NAMES: list[str] = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

CLASS_DESCRIPTIONS: dict[str, str] = {
    "N": "Normal or bundle branch block beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion of ventricular and normal beat",
    "Q": "Unclassifiable or paced beat",
}

# Rare classes. Augmentation hits these harder, because oversampling alone just
# shows the model the same few hundred beats over and over.
MINORITY_CLASSES: tuple[str, ...] = ("S", "V", "F")

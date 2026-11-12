"""Level 1 - the entity label scheme.

Every piece of PII the model learns to find gets exactly one label.
These names are used everywhere later: in the synthetic documents
(Level 2), the BIO tags (Level 3), and the evaluation tables (Level 8).

Keep the scheme small and stable. Renaming a label after Level 2
means regenerating and re-annotating the whole dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityType:
    label: str            # the tag used in the dataset, e.g. "AU_TFN"
    description: str      # what it is, in plain words
    has_checksum: bool    # True -> we can verify it mathematically
    example: str          # a publicly documented / synthetic example


ENTITY_TYPES: tuple[EntityType, ...] = (
    # --- Government / business identifiers with a checksum -------------
    EntityType("AU_TFN", "Tax File Number (ATO), 9 digits, mod-11 checksum",
               True, "123 456 782"),
    EntityType("AU_ABN", "Australian Business Number, 11 digits, mod-89 checksum",
               True, "51 824 753 556"),
    EntityType("AU_ACN", "Australian Company Number, 9 digits, mod-10 check digit",
               True, "004 085 616"),
    EntityType("AU_MEDICARE", "Medicare card number, 10 digits (+ optional IRN)",
               True, "2123 45670 1"),

    # --- Format-only identifiers (no public checksum) -------------------
    EntityType("AU_BSB", "Bank-State-Branch code, 6 digits",
               False, "062-000"),
    EntityType("AU_BANK_ACCOUNT", "Bank account number, 6-10 digits",
               False, "1234 5678"),
    EntityType("AU_PASSPORT", "Australian passport number, 1-2 letters + 7 digits",
               False, "PA1234567"),
    EntityType("AU_DRIVER_LICENCE", "State driver licence number, format varies by state",
               False, "123456789"),
    EntityType("AU_PHONE", "Australian mobile or landline number",
               False, "0412 345 678"),
    EntityType("AU_ADDRESS", "Street address with suburb, state and postcode",
               False, "12 Smith Street, Richmond VIC 3121"),

    # --- General PII ------------------------------------------------------
    EntityType("PERSON", "A person's name", False, "Priya Nguyen"),
    EntityType("EMAIL", "Email address", False, "priya.nguyen@example.com"),
    EntityType("DATE_OF_BIRTH", "Date of birth (Australian DD/MM/YYYY order)",
               False, "03/04/1990"),
)

LABELS: tuple[str, ...] = tuple(e.label for e in ENTITY_TYPES)
BY_LABEL: dict[str, EntityType] = {e.label: e for e in ENTITY_TYPES}
CHECKSUM_LABELS: tuple[str, ...] = tuple(e.label for e in ENTITY_TYPES if e.has_checksum)

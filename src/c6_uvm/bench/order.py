from __future__ import annotations

AGENT_ROLES: dict[str, str] = {
    "subtype": "A_mol",
    "pathway": "A_kg",
    "pathology": "A_path",
    "therapy": "A_tx",
    "router": "A_rt",
    "conformal": "A_cp",
}

_PRECEDENCE: tuple[tuple[str, str], ...] = (
    ("subtype", "pathway"),
    ("pathway", "pathology"),
    ("pathway", "therapy"),
    ("pathology", "router"),
    ("therapy", "router"),
    ("router", "conformal"),
)


def precedence_pairs() -> tuple[tuple[str, str], ...]:
    return _PRECEDENCE


def canonical_order() -> tuple[str, ...]:
    return ("subtype", "pathway", "pathology", "therapy", "router", "conformal")


def validate_schedule(schedule: list[str]) -> bool:
    position = {name: i for i, name in enumerate(schedule)}
    for earlier, later in _PRECEDENCE:
        if earlier in position and later in position and position[earlier] >= position[later]:
            return False
    return True

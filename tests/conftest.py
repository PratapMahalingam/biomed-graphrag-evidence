import pytest

from evidence_rag.data import Claim, Doc


@pytest.fixture
def tiny_docs():
    return [
        Doc(10, "Aspirin and platelets", ("Aspirin inhibits platelet aggregation.", "It acts on cox1 enzymes.")),
        Doc(11, "Metformin in diabetes", ("Metformin lowers glucose.", "It activates ampk signalling.")),
        Doc(12, "AMPK and ageing", ("ampk signalling extends lifespan in worms.", "Mechanisms are unclear.")),
        Doc(13, "Statins", ("Statins reduce cholesterol.", "They inhibit hmg coa reductase.")),
    ]


@pytest.fixture
def tiny_claim():
    return Claim(1, "Metformin extends lifespan through ampk signalling.", {12: {0}}, {12: [(0,)]}, "SUPPORT")

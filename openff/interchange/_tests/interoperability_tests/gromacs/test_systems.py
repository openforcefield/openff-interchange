from openff.toolkit import Quantity

from openff.interchange._tests._gromacs import gmx_roundtrip


def test_ligand_vacuum(caffeine, sage_unconstrained, monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

    topology = caffeine.to_topology()
    topology.box_vectors = Quantity([4, 4, 4], "nanometer")

    gmx_roundtrip(sage_unconstrained.create_interchange(topology))


def test_water_dimer(water_dimer, tip3p, monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

    gmx_roundtrip(tip3p.create_interchange(water_dimer))


def test_alanine_dipeptide(alanine_dipeptide, ff14sb, monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

    gmx_roundtrip(ff14sb.create_interchange(alanine_dipeptide))

from openff.interchange.models import PotentialKey, TopologyKey, VirtualSiteKey


def test_potentialkey_hash_uniqueness():
    """Test that PotentialKey hashes differ when optional attributes are set."""

    smirks = "[#1:1]-[#8X2:2]"
    ref = PotentialKey(id=smirks)
    with_mult = PotentialKey(id=smirks, mult=2)
    with_associated_handler = PotentialKey(id=smirks, associated_handler="espaloma")
    with_bond_order = PotentialKey(id=smirks, bond_order=5 / 4)

    keys = [ref, with_mult, with_associated_handler, with_bond_order]
    assert len({hash(k) for k in keys}) == len(keys)


def test_topologykey_hash_uniqueness():
    """Test that TopologyKey hashes differ when optional attributes are set."""

    smirks = "[#1:1]-[#8X2:2]"
    ref = TopologyKey(id=smirks)
    with_atom_indices = TopologyKey(id=smirks, atom_indices=(2, 0))
    with_mult = TopologyKey(id=smirks, mult=2)
    with_bond_order = TopologyKey(id=smirks, bond_order=5 / 4)

    keys = [ref, with_atom_indices, with_mult, with_bond_order]
    assert len({hash(k) for k in keys}) == len(keys)


def test_virtualsitekey_hash_uniqueness():
    """Test that VirtualSiteKey hashes differ when optional attributes are set."""

    ref = VirtualSiteKey(
        orientation_atom_indices=(0, 1),
        name="vs1",
        type="BondCharge",
        match="once",
    )
    with_name = VirtualSiteKey(
        orientation_atom_indices=(0, 1),
        name="vs2",
        type="TrivalentLonePair",
        match="once",
    )
    with_type = VirtualSiteKey(
        orientation_atom_indices=(0, 1),
        name="vs1",
        type="TrivalentLonePair",
        match="once",
    )
    with_match = VirtualSiteKey(
        orientation_atom_indices=(0, 1),
        name="vs1",
        type="BondCharge",
        match="all_permutations",
    )

    keys = [ref, with_type, with_match, with_name]
    assert len({hash(k) for k in keys}) == len(keys)


def test_reprs():

    topology_key = TopologyKey(atom_indices=(10,))

    assert "atom indices (10,)" in repr(topology_key)
    assert "bond order" not in repr(topology_key)
    assert "mult" not in repr(topology_key)

    topology_key = TopologyKey(atom_indices=(0, 1), mult=2, bond_order=1.111)

    assert "atom indices (0, 1)" in repr(topology_key)
    assert "mult 2" in repr(topology_key)
    assert "bond order 1.111" in repr(topology_key)

    potential_key = PotentialKey(id="foobar")

    assert "foobar" in repr(potential_key)
    assert "mult" not in repr(potential_key)
    assert "bond order" not in repr(potential_key)

    potential_key = PotentialKey(id="blah", mult=2, bond_order=1.111)

    assert "blah" in repr(potential_key)
    assert "mult 2" in repr(potential_key)
    assert "bond order 1.111" in repr(potential_key)

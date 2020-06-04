def build_slots_parameter_map(topology, forcefield):
    big_map = dict()

    for handler_name in ['vdW']:
        slot_potential_map = dict()

        matches = forcefield.get_parameter_handler(handler_name).find_matches(topology)

        for atom_key, atom_match in  matches.items():
            slot_potential_map[atom_key] = atom_match.parameter_type.smirks

        big_map[handler_name] = slot_potential_map

    return big_map

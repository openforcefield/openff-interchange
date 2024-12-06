from pathlib import Path


def make_monolithic(topology_file: str, keep_file: bool = False) -> list[str]:
    parent = Path(topology_file).parent
    name = Path(topology_file).name

    monolithic_path = parent / f"MONOLITHIC_{name}"

    return_value = list()

    with open(topology_file) as in_obj, open(monolithic_path, "w") as out_obj:
        for line in in_obj:
            if len(line) == 0:
                continue

            stripped = line.strip().split()

            if len(stripped) == 0:
                continue

            if stripped[0].startswith(";"):
                continue

            if stripped[0].startswith("#include"):
                itp_file = stripped[1].strip('"')

                with open(itp_file) as itp_obj:
                    for itp_line in itp_obj:
                        if len(itp_line) == 0:
                            continue

                        if itp_line.startswith(";"):
                            continue

                        # TODO: itp files can load from itp files, this block should recurse

                        return_value.append(itp_line)
                        out_obj.write(itp_line)

            else:
                return_value.append(line)
                out_obj.write(line)

    if Path(monolithic_path).exists() and not keep_file:
        Path(monolithic_path).unlink()

    return return_value

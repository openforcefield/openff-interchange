# Export timing benchmarks

Run with the shell script

Example usage:

```console
$ sh run.sh
Running benchmarks for version 0.4.8.post3+ge7f66522
CSV file already exists, exiting
CSV file already exists, exiting
Mixed solvent benchmark summary:
    OpenMM export time (s): 2.734 ± 0.046
    GROMACS export time (s): 0.186 ± 0.065

Ligand in water benchmark summary:
    OpenMM export time (s): 4.212 ± 0.185
    GROMACS export time (s): 1.707 ± 0.141

Compare to most recent release (0.4.8):
Mixed solvent benchmark summary:
    OpenMM export time (s): 2.773 ± 0.066
    GROMACS export time (s): 0.182 ± 0.070

Ligand in water benchmark summary:
    OpenMM export time (s): 4.215 ± 0.155
    GROMACS export time (s): 1.705 ± 0.144

```

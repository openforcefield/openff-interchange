python -m pip install git+https://github.com/openforcefield/interchange-regression-testing.git@443480732f5caf3fc63f5442fdd763176c63e72f

create_openmm_systems \
    --input             "regression_tests/small-molecule/input-topologies.json" \
    --output            "regression_tests/small-molecule/" \
    --using-interchange \
    --force-field       "openff-2.0.0.offxml" \
    --n-procs           4

# Don't trust the interchange version here, for some reason, just put it in a new directory
mkdir regression_tests/small-molecule/omm-systems-interchange-latest/
mv regression_tests/small-molecule/omm-systems-interchange-*/*xml regression_tests/small-molecule/omm-systems-interchange-latest/

compare_openmm_systems \
    --input-dir-a       "regression_tests/small-molecule/omm-systems-toolkit-0.10.6" \
    --input-dir-b       "regression_tests/small-molecule/omm-systems-interchange-latest" \
    --output            "regression_tests/differences.json" \
    --settings          "regression_tests/default-comparison-settings.json" \
    --expected-changes  "regression_tests/toolkit-to-interchange.json" \
    --n-procs           4

python devtools/scripts/molecule-regressions.py

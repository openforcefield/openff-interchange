python -c "import os, openff.interchange; os.makedirs(f'{openff.interchange.__version__}', exist_ok=True); print(f'Running benchmarks for version {openff.interchange.__version__}')"

python mixed-solvent.py
python ligand-in-water.py
python large-peg.py

python summarize.py

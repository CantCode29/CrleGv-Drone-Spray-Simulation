# CrleGV drone spray simulation

This is the computer model behind my honours project (DIS 481, Belgium Campus iTversity).

CrleGV is a virus that kills the false codling moth, a pest of citrus. Sunlight breaks the
virus down within days, but it lasts longer in shade. This model asks: when a drone sprays
the virus onto a citrus tree, how much of it ends up in the shaded inside of the tree, and
how long does it stay active there?

## What the model does

The model looks at a slice of one citrus tree, 4 m wide and 3 m deep, cut into small 5 cm
squares. It works in three steps:

1. **Light.** It works out how much sunlight reaches each square. Leaves and fruit block the
   light, so the inside of the tree is darker.
2. **Spray.** It follows 10,000 spray droplets from the drone down into the tree and records
   where they land.
3. **Virus.** It works out how much of the virus is still active after 24, 48 and 72 hours.
   The more light a square gets, the faster the virus breaks down.

This is done for 81 drone setups: every mix of three flight heights, three flight speeds,
three rotor air speeds and three droplet sizes.

Every number the model uses is in `src/params.py`, with the study it came from.

## How to run it

```
pip install -r requirements.txt
python -m pytest validation -q
PYTHONPATH=src python src/run.py
PYTHONPATH=src python src/validate.py
PYTHONPATH=src python src/sweep.py
PYTHONPATH=src python src/figures.py
```

In order, these install what the model needs, run the checks, do one quick test run,
compare the model with published measurements, run all 81 drone setups (about 45 minutes)
and draw the figures. Results are saved in the `runs` folder and figures in the `figures`
folder.

On Windows PowerShell, run `$env:PYTHONPATH = "src"` first and leave out `PYTHONPATH=src`
from the lines above.

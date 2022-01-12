# Wordle solver

Utility to determine optimal guesses for Wordle.

Files listing legal guesses and answers are included. These lists were extracted from the source code of the Wordle app.

Requires Python version >= 3.10.

## Usage

```bash
git clone git@github.com:aetimmes/wordle-minimax.git
cd wordle-minimax
# To get usage information:
./wordle.py --help
# To get the minimax of the base case:
./wordle.py
# To pass pre-determined guesses:
./wordle.py boozy humph
```

## Known issues

The current solution for the base case is suboptimal - it yields a worst-case of 5 attempts starting with `raise` and `fonly`, where a guess sequence of `ratio` and `lunes` seems to yield a worst-case of 4 attempts. 

## Disclaimer

This code is licensed with the WTFPL. Use at your own risk.

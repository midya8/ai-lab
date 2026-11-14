"""python -m interplab [--seed 0] [--steps 4000] [--save model.npz]: train, then print the circuit report."""
import argparse
from .train import train_induction
from .model import InterpModel
from .report import circuit_report, format_report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="interplab")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--load", help="skip training; load an InterpModel .npz")
    ap.add_argument("--save", help="save the trained weights to this .npz")
    a = ap.parse_args(argv)
    if a.load:
        model = InterpModel.load(a.load)
    else:
        m, hist = train_induction(seed=a.seed, steps=a.steps, verbose=True)
        model = InterpModel.from_minigpt(m)
        if a.save:
            model.save(a.save)
    print(format_report(circuit_report(model), model.layers, model.H))


if __name__ == "__main__":
    main()

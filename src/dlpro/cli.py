import argparse
import json
from pathlib import Path
import numpy as np
from .data import inventory, inspect_cool, read_window, local_oe


def main():
    p = argparse.ArgumentParser(description='Micro-C data audit and local window preview')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit'); a.add_argument('directory')
    a = sub.add_parser('inspect'); a.add_argument('uri')
    a = sub.add_parser('window')
    a.add_argument('uri'); a.add_argument('chrom'); a.add_argument('start', type=int)
    a.add_argument('end', type=int); a.add_argument('--output', required=True)
    a.add_argument('--balance', action='store_true')
    args = p.parse_args()
    if args.command == 'audit':
        print(json.dumps(inventory(args.directory), indent=2)); return
    if args.command == 'inspect':
        print(json.dumps(inspect_cool(args.uri), indent=2)); return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a, mask = read_window(args.uri, args.chrom, args.start, args.end, args.balance)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    oe = local_oe(a)
    np.savez_compressed(out/'window.npz', counts=a, valid=mask, local_oe=oe)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(np.log1p(oe), origin='lower', cmap='magma', vmin=0, vmax=np.log(5))
    ax.set(title='Exploratory local O/E', xlabel='Local bin', ylabel='Local bin')
    fig.colorbar(im, ax=ax, label='log(1 + local O/E)')
    fig.tight_layout(); fig.savefig(out/'window.png', dpi=160); plt.close(fig)
    (out/'metadata.json').write_text(json.dumps(vars(args), indent=2)+'\n')

if __name__ == '__main__':
    main()

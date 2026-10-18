"""Circular source-bin coordinates must preserve partial-bin width and overlap."""
import importlib.util
from pathlib import Path
import pandas as pd

spec=importlib.util.spec_from_file_location('candidate_analysis',Path(__file__).parents[1]/'scripts/analyze_discovery_candidates.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_circular_scan_window_retains_short_terminal_bin_and_half_open_bounds():
 parts=module.genomic_segments(0)
 assert parts==[(4635300,4641652),(0,6400)]
 assert sum(b-a for a,b in parts)==12752
 known=pd.DataFrame({'start':[4641640,6399,6400,100000],'end':[4641650,6400,6500,101000]})
 assert module.overlaps(parts,known).tolist()==[True,True,False,False]
 assert module.overlaps(parts,known,200).tolist()==[True,True,True,False]
 assert module.edge_distances(parts,known).tolist()==[0,0,0,93600]


def test_localized_bin_interval_uses_same_source_grid_as_scan():
 full=module.genomic_segments(19200)
 local=module.genomic_segments(19200,100,8,24)
 assert full==[(12800,25600)]
 assert local==[(13600,15200)]

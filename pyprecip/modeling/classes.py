import numpy as np
import pandas as pd
from typing import List, Tuple

def make_class_bins(intervals: List[Tuple[float,float]]):
    bin_edges = [a for a,_ in intervals] + [intervals[-1][1]]
    class_means = [(a+b)/2 for (a,b) in intervals]
    return bin_edges, class_means

def to_class_indices(y: pd.Series, bin_edges):
    y_cls = pd.cut(y, bins=bin_edges, right=False, labels=False)
    mask = ~y_cls.isna()
    return y_cls[mask].astype(int).values, mask
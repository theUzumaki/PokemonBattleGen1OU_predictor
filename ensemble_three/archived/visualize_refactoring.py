"""
Visualization utilities for refactoring comparison (archived).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

def create_refactoring_visuals():
    categories = ['Team\nComposition', 'Opponent\nLead', 'Battle\nTimeline', 'Derived\nRatios']
    before_importance = [30.88, 7.49, 33.37, 8.54]
    after_importance = [16.26, 4.25, 52.06, 0.96]
    # Simplified plotting preserved from original
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(categories))
    ax.bar(x - 0.15, before_importance, width=0.3, label='Before')
    ax.bar(x + 0.15, after_importance, width=0.3, label='After')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    out = results_dir / 'refactoring_comparison.png'
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_refactoring_visuals()

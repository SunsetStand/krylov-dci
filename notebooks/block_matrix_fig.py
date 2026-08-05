import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def draw_ci_structure(ax):
    """Draws the detailed CI matrix structure for the CASCI problem."""

    # Plot parameters
    total_len = 100
    n_elec = 10
    total_occ = n_elec # n_A + n_B
    
    # Define occupation blocks (starting y-coordinate and height)
    # The dimensions (height/width) of the blocks should vary.
    # Higher occupation might have larger dimensions.
    # The spacing should reflect the partition.
    
    blocks = []
    current_pos = 0
    
    for n in range(n_elec + 1):
        # Placeholder dimensions - in reality, they depend on nA, nB, dim(FA), dim(FB)
        # We make them monotonic for illustration.
        if n == 0 or n == 10:
            height = 5 
        elif n == 1 or n == 9:
            height = 7
        elif n == 2 or n == 8:
            height = 10
        elif n == 3 or n == 7:
            height = 12
        elif n == 4 or n == 6:
            height = 15
        else: # n=5
            height = 18

        blocks.append({
            'n': n,
            'y_start': current_pos,
            'height': height,
            'x_start': current_pos,
            'width': height
        })
        current_pos += height

    # 1. Main Matrix Frame
    ax.add_patch(Rectangle((0, 0), current_pos, current_pos, linewidth=2, edgecolor='black', facecolor='none'))
    
    # 2. Draw non-zero (A, B) pairs and labels
    # Here, we only have one type of pairs (nA=... nB=...), so we just label n.
    for block in blocks:
        n = block['n']
        ax.add_patch(Rectangle((block['x_start'], block['y_start']), block['width'], block['height'], 
                               linewidth=1, edgecolor='grey', facecolor='lightgrey', alpha=0.5))
        
        # Add labels on the axes
        # ax.text(-10, block['y_start'] + block['height']/2, f"n={n}", va='center', ha='right', fontsize=12, fontweight='bold', family='monospace')
        # ax.text(block['x_start'] + block['width']/2, current_pos + 5, f"n={n}", va='bottom', ha='center', fontsize=12, fontweight='bold', family='monospace')
        
    # 3. Highlight P-space and Q-space
    # P-space indices: n=8, 9, 10
    p_indices = [8, 9, 10]
    p_blocks = [b for b in blocks if b['n'] in p_indices]
    
    q_indices = [n for n in range(n_elec + 1) if n not in p_indices]
    q_blocks = [b for b in blocks if b['n'] in q_indices]
    
    # --- Highlight and Label P-space ---
    for b in p_blocks:
        ax.add_patch(Rectangle((b['x_start'], b['y_start']), b['width'], b['height'], 
                               linewidth=2, edgecolor='red', facecolor='orange', alpha=0.8, hatch='//'))
        
        # Label each block in the P-space
        ax.text(b['x_start'] + b['width']/2, b['y_start'] + b['height']/2, f"$C^{{({b['n']})}}$", 
                va='center', ha='center', fontsize=16, fontweight='bold', color='black', family='serif')

    # Overall P-space brace and label on the left axis
    # The P-space is a continuous range (8-10)
    p_y_start = p_blocks[0]['y_start']
    p_y_end = p_blocks[-1]['y_start'] + p_blocks[-1]['height']
    
    # Bracket on the left for P-space
    bracket_offset = -3
    ax.plot([bracket_offset, 0], [p_y_start, p_y_start], color='red', linewidth=3)
    ax.plot([bracket_offset, bracket_offset], [p_y_start, p_y_end], color='red', linewidth=3)
    ax.plot([bracket_offset, 0], [p_y_end, p_y_end], color='red', linewidth=3)
    ax.text(-12, (p_y_start + p_y_end)/2, "Primary Space (P)", va='center', ha='center', 
            rotation=90, fontsize=16, fontweight='bold', color='red', family='serif',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Label on the left with {8,9,10}
    ax.text(-30, (p_y_start + p_y_end)/2, "$n \in \\{8, 9, 10\\}$", va='center', ha='center', 
            rotation=90, fontsize=14, fontweight='bold', color='red', family='monospace')

    # --- Highlight and Label Q-space ---
    for b in q_blocks:
        ax.add_patch(Rectangle((b['x_start'], b['y_start']), b['width'], b['height'], 
                               linewidth=1.5, edgecolor='blue', facecolor='cyan', alpha=0.6))
        
        # Label each block in the Q-space
        ax.text(b['x_start'] + b['width']/2, b['y_start'] + b['height']/2, f"$C^{{({b['n']})}}$", 
                va='center', ha='center', fontsize=14, fontweight='bold', color='black', family='serif')

    # Overall Q-space labels
    # The Q-space is composed of two discontinuous ranges (0-7 and none here, then n=?)
    # Since they are not continuous, we just label the indices and the type.
    
    # Collect ranges and single values for Q-space indices
    from collections import defaultdict
    def group_indices(indices):
        if not indices: return []
        ranges = []
        start = indices[0]
        end = start
        for i in indices[1:]:
            if i == end + 1:
                end = i
            else:
                ranges.append((start, end))
                start = i
                end = i
        ranges.append((start, end))
        
        # Format the result
        result = []
        for s, e in ranges:
            if s == e: result.append(str(s))
            else: result.append(f"{s}-{e}")
        return result

    q_indices_labeled = group_indices(q_indices)
    
    # Find a good place to label. The continuous block on the left is 0-7.
    q_left_blocks = [b for b in q_blocks if b['n'] <= 7]
    q_l_y_start = q_left_blocks[0]['y_start']
    q_l_y_end = q_left_blocks[-1]['y_start'] + q_left_blocks[-1]['height']

    # Label on the left axis with indices
    ax.text(-30, (q_l_y_start + q_l_y_end)/2, "$n \in \\{" + ", ".join(q_indices_labeled) + "\\}$", va='center', ha='center', 
            rotation=90, fontsize=14, fontweight='bold', color='blue', family='monospace')
    
    # Bracket on the left for Q-space
    bracket_offset = -3
    ax.plot([bracket_offset, 0], [q_l_y_start, q_l_y_start], color='blue', linewidth=3)
    ax.plot([bracket_offset, bracket_offset], [q_l_y_start, q_l_y_end], color='blue', linewidth=3)
    ax.plot([bracket_offset, 0], [q_l_y_end, q_l_y_end], color='blue', linewidth=3)
    
    # Overall label on the left axis
    ax.text(-12, (q_l_y_start + q_l_y_end)/2, "Secondary Space (Q)", va='center', ha='center', 
            rotation=90, fontsize=16, fontweight='bold', color='blue', family='serif',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Label on the top axis as well?
    for n in range(n_elec + 1):
        block = next(b for b in blocks if b['n'] == n)
        color = 'red' if n in p_indices else 'blue'
        # ax.text(block['x_start'] + block['width']/2, current_pos + 10, f"$C^{{({n})}}$", 
        #        va='bottom', ha='center', fontsize=12, fontweight='bold', color=color, family='serif')

    # 4. Legend
    legend_elements = [
        Patch(facecolor='orange', edgecolor='red', hatch='//', label='P-space: $n \in \{8, 9, 10\}$'),
        Patch(facecolor='cyan', edgecolor='blue', label=f'Q-space: $n \in \{{0 \dots 7\}}$')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12, framealpha=0.9, edgecolor='black')

    # 5. Titles and Labels
    ax.set_title("Block-Diagonal Structure of CASCI Coefficient Matrix $C$", fontsize=22, fontweight='bold', family='serif')
    # Use larger text for axes
    # ax.set_xlabel("Basis B (right singular vectors $V^{(n)}$)", fontsize=18)
    # ax.set_ylabel("Basis A (left singular vectors $U^{(n)}$)", fontsize=18)
    ax.set_xticks([]) # Remove tick marks, use labels instead
    ax.set_yticks([]) 
    
    # Total dimension label
    ax.text(current_pos / 2, -8, "Full basis size $D = \\dim(\\mathcal{F}_A \\otimes \\mathcal{F}_B)$", va='center', ha='center', 
            fontsize=16, fontweight='bold', family='monospace')
    ax.text(-44, current_pos / 2, "Full basis size $D$", va='center', ha='center', rotation=90,
            fontsize=16, fontweight='bold', family='monospace')

    ax.axis('off') # Hide default axis, use custom labels

from matplotlib.patches import Patch
import matplotlib.patches as mpatches

# Create figure and axes
fig, ax = plt.subplots(figsize=(16, 12))
# Adjust limits to accommodate labels outside the main area
ax.set_xlim(-50, 125)
ax.set_ylim(-15, 125)
draw_ci_structure(ax)

# Show and save the plot
plt.tight_layout()
plt.show()
fig.savefig('figures/fig8_block_diagonal.png', dpi=300, bbox_inches='tight')

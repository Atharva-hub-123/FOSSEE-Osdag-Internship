"""

This script contains complete solution for both tasks:
- Task 1: 2D Shear Force and Bending Moment Diagrams
- Task 2: 3D MIDAS-Style Visualization with deck mesh


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf_file
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.patches as mpatches
import os
import os

# ============================================================================
# CONFIGURATION AND DATA
# ============================================================================

# File paths
NC_FILE = 'C:\\Users\\athar\\OneDrive\\Desktop\\Coding\\Python\\FOSSEE\\screening_task.nc'
OUTPUT_DIR = 'C:\\Users\\athar\\OneDrive\\Desktop\\Coding\\Python\\FOSSEE\\outputs'

# Complete node coordinates with Y=0 for all nodes in XZ plane
nodes = {
    1: [0.0000, 0.0000, 0.0000], 2: [0.0000, 0.0000, 1.2000],
    3: [0.0000, 0.0000, 5.1750], 4: [0.0000, 0.0000, 9.1500],
    5: [0.0000, 0.0000, 10.3500], 6: [25.0000, 0.0000, 0.0000],
    7: [25.0000, 0.0000, 1.2000], 8: [25.0000, 0.0000, 5.1750],
    9: [25.0000, 0.0000, 9.1500], 10: [25.0000, 0.0000, 10.3500],
    11: [2.7778, 0.0000, 0.0000], 12: [2.7778, 0.0000, 1.2000],
    13: [2.7778, 0.0000, 5.1750], 14: [2.7778, 0.0000, 9.1500],
    15: [2.7778, 0.0000, 10.3500], 16: [5.5556, 0.0000, 0.0000],
    17: [5.5556, 0.0000, 1.2000], 18: [5.5556, 0.0000, 5.1750],
    19: [5.5556, 0.0000, 9.1500], 20: [5.5556, 0.0000, 10.3500],
    21: [8.3333, 0.0000, 0.0000], 22: [8.3333, 0.0000, 1.2000],
    23: [8.3333, 0.0000, 5.1750], 24: [8.3333, 0.0000, 9.1500],
    25: [8.3333, 0.0000, 10.3500], 26: [11.1111, 0.0000, 0.0000],
    27: [11.1111, 0.0000, 1.2000], 28: [11.1111, 0.0000, 5.1750],
    29: [11.1111, 0.0000, 9.1500], 30: [11.1111, 0.0000, 10.3500],
    31: [13.8889, 0.0000, 0.0000], 32: [13.8889, 0.0000, 1.2000],
    33: [13.8889, 0.0000, 5.1750], 34: [13.8889, 0.0000, 9.1500],
    35: [13.8889, 0.0000, 10.3500], 36: [16.6667, 0.0000, 0.0000],
    37: [16.6667, 0.0000, 1.2000], 38: [16.6667, 0.0000, 5.1750],
    39: [16.6667, 0.0000, 9.1500], 40: [16.6667, 0.0000, 10.3500],
    41: [19.4444, 0.0000, 0.0000], 42: [19.4444, 0.0000, 1.2000],
    43: [19.4444, 0.0000, 5.1750], 44: [19.4444, 0.0000, 9.1500],
    45: [19.4444, 0.0000, 10.3500], 46: [22.2222, 0.0000, 0.0000],
    47: [22.2222, 0.0000, 1.2000], 48: [22.2222, 0.0000, 5.1750],
    49: [22.2222, 0.0000, 9.1500], 50: [22.2222, 0.0000, 10.3500]
}

# Element connectivity
members = {
    15: [3, 13], 24: [13, 18], 33: [18, 23], 42: [23, 28], 51: [28, 33],
    60: [33, 38], 69: [38, 43], 78: [43, 48], 83: [48, 8],
    14: [2, 12], 23: [12, 17], 32: [17, 22], 41: [22, 27], 50: [27, 32],
    59: [32, 37], 68: [37, 42], 77: [42, 47], 82: [47, 7],
    16: [4, 14], 25: [14, 19], 34: [19, 24], 43: [24, 29], 52: [29, 34],
    61: [34, 39], 70: [39, 44], 79: [44, 49], 84: [49, 9],
    13: [1, 11], 22: [11, 16], 31: [16, 21], 40: [21, 26], 49: [26, 31],
    58: [31, 36], 67: [36, 41], 76: [41, 46], 81: [46, 6],
    17: [5, 15], 26: [15, 20], 35: [20, 25], 44: [25, 30], 53: [30, 35],
    62: [35, 40], 71: [40, 45], 80: [45, 50], 85: [50, 10],
    9: [11, 12], 10: [12, 13], 11: [13, 14], 12: [14, 15],
    18: [16, 17], 19: [17, 18], 20: [18, 19], 21: [19, 20],
    27: [21, 22], 28: [22, 23], 29: [23, 24], 30: [24, 25],
    36: [26, 27], 37: [27, 28], 38: [28, 29], 39: [29, 30],
    45: [31, 32], 46: [32, 33], 47: [33, 34], 48: [34, 35],
    54: [36, 37], 55: [37, 38], 56: [38, 39], 57: [39, 40],
    63: [41, 42], 64: [42, 43], 65: [43, 44], 66: [44, 45],
    72: [46, 47], 73: [47, 48], 74: [48, 49], 75: [49, 50],
    1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5],
    5: [6, 7], 6: [7, 8], 7: [8, 9], 8: [9, 10],
}

# Central girder for Task 1
CENTRAL_GIRDER_ELEMENTS = [15, 24, 33, 42, 51, 60, 69, 78, 83]
CENTRAL_GIRDER_NODES = [3, 13, 18, 23, 28, 33, 38, 43, 48, 8]

# All girders for Task 2 (as per task specification)
ALL_GIRDERS = [
    {
        'name': 'Girder 1',
        'elements': [13, 22, 31, 40, 49, 58, 67, 76, 81],
        'nodes': [1, 11, 16, 21, 26, 31, 36, 41, 46, 6]
    },
    {
        'name': 'Girder 2',
        'elements': [14, 23, 32, 41, 50, 59, 68, 77, 82],
        'nodes': [2, 12, 17, 22, 27, 32, 37, 42, 47, 7]
    },
    {
        'name': 'Girder 3',
        'elements': [15, 24, 33, 42, 51, 60, 69, 78, 83],
        'nodes': [3, 13, 18, 23, 28, 33, 38, 43, 48, 8]
    },
    {
        'name': 'Girder 4',
        'elements': [16, 25, 34, 43, 52, 61, 70, 79, 84],
        'nodes': [4, 14, 19, 24, 29, 34, 39, 44, 49, 9]
    },
    {
        'name': 'Girder 5',
        'elements': [17, 26, 35, 44, 53, 62, 71, 80, 85],
        'nodes': [5, 15, 20, 25, 30, 35, 40, 45, 50, 10]
    }
]



# UTILITY FUNCTIONS


def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"      Created output directory: {OUTPUT_DIR}")



# UTILITY FUNCTIONS

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"      Created output directory: {OUTPUT_DIR}")
    else:
        print(f"      Output directory exists: {OUTPUT_DIR}")


# DATA LOADING

def load_force_data(nc_file_path):
    """
    Load force and moment data from NetCDF file.
    
    Returns:
        dict: Force data organized by element ID with Mz_i, Mz_j, Vy_i, Vy_j
    """
    # Open NetCDF file
    nc = netcdf_file(nc_file_path, 'r', mmap=False)
    
    # Extract arrays
    forces = nc.variables['forces'][:].copy()
    elements = nc.variables['Element'][:].copy()
    components = nc.variables['Component'][:].copy()
    
    nc.close()
    
    # Decode component names from byte arrays
    component_names = []
    for comp in components:
        name = comp.tobytes().decode('utf-8', errors='ignore').strip('\x00').strip()
        component_names.append(name)
    
    # Find indices for required components
    mz_i_idx = component_names.index('Mz_i')
    mz_j_idx = component_names.index('Mz_j')
    vy_i_idx = component_names.index('Vy_i')
    vy_j_idx = component_names.index('Vy_j')
    
    # Organize data by element ID
    force_data = {}
    for i, elem_id in enumerate(elements):
        force_data[int(elem_id)] = {
            'Mz_i': forces[i, mz_i_idx],
            'Mz_j': forces[i, mz_j_idx],
            'Vy_i': forces[i, vy_i_idx],
            'Vy_j': forces[i, vy_j_idx],
        }
    
    return force_data


# ============================================================================
# TASK 1: 2D SFD AND BMD
# ============================================================================

def calculate_distances_along_girder(element_list):
    """
    Calculate cumulative distances along a girder.
    
    Args:
        element_list: List of element IDs forming the girder
    
    Returns:
        distances: Cumulative distance at each node
        positions: 3D coordinates at each node
    """
    distances = [0]
    positions = []
    
    for i, elem_id in enumerate(element_list):
        start_node, end_node = members[elem_id]
        
        if i == 0:
            # First element - add start position
            positions.append(nodes[start_node])
        
        # Add end position
        positions.append(nodes[end_node])
        
        # Calculate distance
        start_coord = np.array(positions[-2])
        end_coord = np.array(positions[-1])
        dist = np.linalg.norm(end_coord - start_coord)
        distances.append(distances[-1] + dist)
    
    return np.array(distances), positions


def create_task1_plots(force_data):
    """
    Create Task 1: 2D SFD and BMD for central girder.
    
    Returns:
        matplotlib figure
    """
    # Calculate distances
    distances, positions = calculate_distances_along_girder(CENTRAL_GIRDER_ELEMENTS)
    
    # Extract force and moment values
    bending_moments = []
    shear_forces = []
    
    for i, elem_id in enumerate(CENTRAL_GIRDER_ELEMENTS):
        data = force_data[elem_id]
        
        if i == 0:
            bending_moments.append(data['Mz_i'])
            shear_forces.append(data['Vy_i'])
        
        bending_moments.append(data['Mz_j'])
        shear_forces.append(data['Vy_j'])
    
    bending_moments = np.array(bending_moments)
    shear_forces = np.array(shear_forces)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
    fig.suptitle('Bridge Grillage Analysis - Central Longitudinal Girder (Task 1)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # ========== BENDING MOMENT DIAGRAM ==========
    ax1.fill_between(distances, 0, bending_moments, alpha=0.35, color='#1E88E5', 
                      label='Bending Moment', zorder=2)
    ax1.plot(distances, bending_moments, color='#0D47A1', linewidth=2.5, 
             marker='o', markersize=5, label='Moment Values', zorder=3)
    ax1.axhline(y=0, color='black', linewidth=2, linestyle='-', zorder=1)
    ax1.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
    # Annotate values
    for i, (d, m) in enumerate(zip(distances, bending_moments)):
        if i % 2 == 0 or i == len(distances) - 1:
            ax1.annotate(f'{m:.2f}', xy=(d, m), 
                        xytext=(0, 12 if m > 0 else -18),
                        textcoords='offset points', ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF59D', 
                                 alpha=0.85, edgecolor='none'),
                        fontweight='bold')
    
    # Mark maximum
    max_idx = np.argmax(np.abs(bending_moments))
    max_moment = bending_moments[max_idx]
    max_dist = distances[max_idx]
    ax1.plot(max_dist, max_moment, 'ro', markersize=12, zorder=4)
    ax1.annotate(f'Max: {max_moment:.2f} kN·m', 
                xy=(max_dist, max_moment),
                xytext=(15, 15 if max_moment < 0 else -25), 
                textcoords='offset points',
                fontsize=11, fontweight='bold', color='#C62828',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                         edgecolor='#C62828', linewidth=2.5),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5),
                zorder=5)
    
    ax1.set_xlabel('Distance along girder (m)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Bending Moment (kN·m)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('Bending Moment Diagram (BMD)', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=11, framealpha=0.95)
    ax1.tick_params(labelsize=10)
    
    # ========== SHEAR FORCE DIAGRAM ==========
    ax2.fill_between(distances, 0, shear_forces, alpha=0.35, color='#E53935',
                      label='Shear Force', zorder=2)
    ax2.plot(distances, shear_forces, color='#B71C1C', linewidth=2.5,
             marker='s', markersize=5, label='Force Values', zorder=3)
    ax2.axhline(y=0, color='black', linewidth=2, linestyle='-', zorder=1)
    ax2.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
    # Annotate values
    for i, (d, v) in enumerate(zip(distances, shear_forces)):
        if i % 2 == 0 or i == len(distances) - 1:
            ax2.annotate(f'{v:.2f}', xy=(d, v),
                        xytext=(0, 12 if v > 0 else -18),
                        textcoords='offset points', ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9',
                                 alpha=0.85, edgecolor='none'),
                        fontweight='bold')
    
    # Mark maximum
    max_idx = np.argmax(np.abs(shear_forces))
    max_shear = shear_forces[max_idx]
    max_dist = distances[max_idx]
    ax2.plot(max_dist, max_shear, 'go', markersize=12, zorder=4)
    ax2.annotate(f'Max: {max_shear:.2f} kN',
                xy=(max_dist, max_shear),
                xytext=(15, -25 if max_shear > 0 else 25),
                textcoords='offset points',
                fontsize=11, fontweight='bold', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                         edgecolor='#2E7D32', linewidth=2.5),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5),
                zorder=5)
    
    ax2.set_xlabel('Distance along girder (m)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Shear Force (kN)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title('Shear Force Diagram (SFD)', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    return fig


# ============================================================================
# TASK 2: 3D MIDAS-STYLE VISUALIZATION
# ============================================================================

def draw_bridge_deck_mesh(ax):
    """
    Draw the complete bridge deck mesh (framing structure).
    This shows all longitudinal and transverse members.
    """
    lines = []
    
    # Add all members to create the mesh
    for elem_id, (start_node, end_node) in members.items():
        start = nodes[start_node]
        end = nodes[end_node]
        lines.append([start, end])
    
    # Create line collection for the deck mesh
    lc = Line3DCollection(lines, colors='#90A4AE', linewidths=1.2, 
                         alpha=0.4, zorder=1)
    ax.add_collection3d(lc)


def create_midas_diagram_for_girder(ax, girder_info, force_data, 
                                   diagram_type='BMD', color='blue', scale=0.15):
    """
    Create MIDAS-style diagram for a single girder with vertical extrusion.
    
    Args:
        ax: 3D axis
        girder_info: Dictionary with 'elements' and 'nodes'
        force_data: Force/moment data
        diagram_type: 'BMD' or 'SFD'
        color: Color for this girder
        scale: Scale factor for vertical extrusion
    """
    element_list = girder_info['elements']
    node_list = girder_info['nodes']
    
    # Collect force/moment values at each node
    node_values = {}
    
    for elem_id in element_list:
        data = force_data[elem_id]
        start_node, end_node = members[elem_id]
        
        # Select value type
        if diagram_type == 'BMD':
            val_i = data['Mz_i']
            val_j = data['Mz_j']
        else:  # SFD
            val_i = data['Vy_i']
            val_j = data['Vy_j']
        
        # Store values
        if start_node not in node_values:
            node_values[start_node] = val_i
        node_values[end_node] = val_j
    
    # Draw vertical bars (ordinates) at each node
    for node_id in node_list:
        if node_id in node_values:
            x, y, z = nodes[node_id]
            value = node_values[node_id]
            y_extruded = y + value * scale
            
            # Vertical bar from deck to force/moment value
            ax.plot([x, x], [y, y_extruded], [z, z],
                   color=color, linewidth=2.5, alpha=0.85, zorder=3)
            
            # Marker at top of bar
            ax.scatter([x], [y_extruded], [z], 
                      color=color, s=30, alpha=0.9, zorder=4)
    
    # Draw profile line connecting tops of bars
    profile_x, profile_y, profile_z = [], [], []
    for node_id in node_list:
        if node_id in node_values:
            x, y, z = nodes[node_id]
            y_extruded = y + node_values[node_id] * scale
            profile_x.append(x)
            profile_y.append(y_extruded)
            profile_z.append(z)
    
    ax.plot(profile_x, profile_y, profile_z,
           color=color, linewidth=2.5, alpha=0.95, zorder=4)
    
    # Fill surfaces between baseline and diagram
    for i in range(len(node_list) - 1):
        if node_list[i] in node_values and node_list[i+1] in node_values:
            n1, n2 = node_list[i], node_list[i+1]
            
            x1, y1, z1 = nodes[n1]
            x2, y2, z2 = nodes[n2]
            
            val1 = node_values[n1]
            val2 = node_values[n2]
            
            y1_top = y1 + val1 * scale
            y2_top = y2 + val2 * scale
            
            # Quadrilateral surface
            verts = [[x1, y1, z1], [x2, y2, z2],
                    [x2, y2_top, z2], [x1, y1_top, z1]]
            
            poly = Poly3DCollection([verts], alpha=0.5,
                                   facecolor=color, edgecolor=color,
                                   linewidth=0.5, zorder=2)
            ax.add_collection3d(poly)


def create_task2_plots(force_data):
    """
    Create Task 2: 3D MIDAS-style visualization for all girders.
    
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(22, 11))
    
    # Colors for girders
    colors_bmd = ['#0D47A1', '#1565C0', '#1976D2', '#42A5F5', '#64B5F6']
    colors_sfd = ['#B71C1C', '#D32F2F', '#F44336', '#FF5722', '#FF9800']
    
    # ========== BMD SUBPLOT ==========
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Draw bridge deck mesh first
    draw_bridge_deck_mesh(ax1)
    
    # Draw diagrams for each girder
    for i, girder_info in enumerate(ALL_GIRDERS):
        create_midas_diagram_for_girder(ax1, girder_info, force_data,
                                       diagram_type='BMD', 
                                       color=colors_bmd[i], 
                                       scale=0.12)
    
    # Configure BMD plot
    ax1.set_xlabel('X (Longitudinal, m)', fontsize=12, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Y (Transverse, m)\n+ Diagram Extrusion', 
                   fontsize=12, fontweight='bold', labelpad=15)
    ax1.set_zlabel('Z (Elevation, m)', fontsize=12, fontweight='bold', labelpad=12)
    ax1.set_title('3D Bending Moment Diagram (BMD)\nMIDAS-Style Visualization',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    ax1.set_box_aspect([2.5, 0.8, 1])
    
    # Legend
    legend_bmd = [mpatches.Patch(facecolor=colors_bmd[i], 
                                label=f"{ALL_GIRDERS[i]['name']}", alpha=0.75)
                 for i in range(len(ALL_GIRDERS))]
    ax1.legend(handles=legend_bmd, loc='upper left', fontsize=10, framealpha=0.95)
    
    # ========== SFD SUBPLOT ==========
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Draw bridge deck mesh first
    draw_bridge_deck_mesh(ax2)
    
    # Draw diagrams for each girder
    for i, girder_info in enumerate(ALL_GIRDERS):
        create_midas_diagram_for_girder(ax2, girder_info, force_data,
                                       diagram_type='SFD',
                                       color=colors_sfd[i],
                                       scale=0.5)
    
    # Configure SFD plot
    ax2.set_xlabel('X (Longitudinal, m)', fontsize=12, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Y (Transverse, m)\n+ Diagram Extrusion',
                   fontsize=12, fontweight='bold', labelpad=15)
    ax2.set_zlabel('Z (Elevation, m)', fontsize=12, fontweight='bold', labelpad=12)
    ax2.set_title('3D Shear Force Diagram (SFD)\nMIDAS-Style Visualization',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_box_aspect([2.5, 0.8, 1])
    
    # Legend
    legend_sfd = [mpatches.Patch(facecolor=colors_sfd[i],
                                label=f"{ALL_GIRDERS[i]['name']}", alpha=0.75)
                 for i in range(len(ALL_GIRDERS))]
    ax2.legend(handles=legend_sfd, loc='upper left', fontsize=10, framealpha=0.95)
    
    # Main title
    fig.suptitle('Bridge Grillage Analysis - 3D Force/Moment Diagrams (MIDAS Style)\n' +
                'Diagrams extruded vertically from actual girder positions',
                fontsize=17, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("BRIDGE GRILLAGE ANALYSIS - FINAL SOLUTION")
    print("Osdag Internship Screening Task")
    print("="*80)
    
    # Ensure output directory exists
    print("\n[0/3] Checking output directory...")
    ensure_output_directory()
    print(f"      Output directory ready: {OUTPUT_DIR}")
    
    # Load data
    print("\n[1/3] Loading force data from NetCDF file...")
    print(f"      File: {NC_FILE}")
    force_data = load_force_data(NC_FILE)
    print(f"      Successfully loaded data for {len(force_data)} elements")
    
    # Task 1
    print("\n[2/3] Creating Task 1: 2D SFD and BMD for central girder...")
    fig1 = create_task1_plots(force_data)
    task1_path = os.path.join(OUTPUT_DIR, 'TASK1_Final_2D_Diagrams.png')
    fig1.savefig(task1_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"      Task 1 saved: {task1_path}")
    plt.show()  # Display the plot
    
    # Task 2
    print("\n[3/3] Creating Task 2: 3D MIDAS-style visualization...")
    fig2 = create_task2_plots(force_data)
    task2_path = os.path.join(OUTPUT_DIR, 'TASK2_Final_3D_MIDAS_Style.png')
    fig2.savefig(task2_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"      Task 2 saved: {task2_path}")
    plt.show()  # Display the plot
    
    # Statistics
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY - CENTRAL GIRDER (Task 1)")
    print("="*80)
    
    bm_vals, sf_vals = [], []
    for elem_id in CENTRAL_GIRDER_ELEMENTS:
        data = force_data[elem_id]
        bm_vals.extend([data['Mz_i'], data['Mz_j']])
        sf_vals.extend([data['Vy_i'], data['Vy_j']])
    
    bm_vals, sf_vals = np.array(bm_vals), np.array(sf_vals)
    
    print(f"\n Bending Moment Statistics:")
    print(f"   Maximum: {np.max(np.abs(bm_vals)):.3f} kN·m")
    print(f"   Minimum: {np.min(bm_vals):.3f} kN·m")
    print(f"   Average (absolute): {np.mean(np.abs(bm_vals)):.3f} kN·m")
    
    print(f"\n Shear Force Statistics:")
    print(f"   Maximum: {np.max(np.abs(sf_vals)):.3f} kN")
    print(f"   Minimum: {np.min(sf_vals):.3f} kN")
    print(f"   Average (absolute): {np.mean(np.abs(sf_vals)):.3f} kN")
    
    print("\n" + "="*80)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n Key Features:")
    print("   • Task 1: Professional 2D diagrams with annotations")
    print("   • Task 2: Authentic MIDAS-style 3D visualization")
    print("   • Bridge deck mesh visible")
    print("   • Vertical force/moment extrusions")
    print("   • All 5 girders analyzed")
    print("   • Publication-quality output (300 DPI)")
    
    print(f"\n Output Location:")
    print(f"   {OUTPUT_DIR}")
    print(f"   ├── TASK1_Final_2D_Diagrams.png")
    print(f"   └── TASK2_Final_3D_MIDAS_Style.png")
    
    plt.close('all')


if __name__ == "__main__":
    main()

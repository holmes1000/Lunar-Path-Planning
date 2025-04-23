"""
Samara Holmes
Spring 2025

Initial coverage path planning program

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import random
    
class CoveragePathPlanner:
    def __init__(self, grid_map, tool_size=1):
        """
        Initialize the coverage path planner.
        
        Args:
            grid_map: 2D binary numpy array where 1 indicates area to be covered, 0 indicates obstacles or areas to avoid
            tool_size: Size of the robot's square tool (in grid cells)
        """
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        self.tool_size = tool_size
        
        # Create cell map and grid graph
        self.cell_map = self._create_cell_map()
        self.grid_graph = self._create_grid_graph()
        
        # Classification of cells
        self.trunk_cells = set()  # Complete mega-cells
        self.branch_cells = set()  # Regular cells not part of complete mega-cells
        self.subsidiary_cells = set()  # Occupied cells, cells with multiple path points, etc.
        
        # Path points
        self.path_points = {}  # Dictionary mapping cell coordinates to path point coordinates
        
    def _create_cell_map(self):
        """
        Create a cell map from the region map.
        0 in the grid map is a 1 in the cell map
        """
        cell_map = np.zeros((self.height, self.width), dtype=int)
        
        for i in range(self.height):
            for j in range(self.width):
                if self.grid_map[i, j] == 0: # IF GRID_MAP IS 0 IT MEANS THAT IT NEEDS TO BE COVERED
                    cell_map[i, j] = 1
        return cell_map
    
    def _create_grid_graph(self):
        """Create a grid graph from the cell map."""
        grid_graph = nx.Graph()
        
        # Add nodes for each cell in the cell map
        for i in range(self.height):
            for j in range(self.width):
                if self.cell_map[i, j] == 1:
                    grid_graph.add_node((i, j))
        
        # Add edges between adjacent cells
        # for i in range(self.height):
        #     for j in range(self.width):
        #         if self.cell_map[i, j] == 1:
        #             # Check neighbors
        #             for ni, nj in [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]:  # Added all four directions
        #                 if 0 <= ni < self.height and 0 <= nj < self.width and self.cell_map[ni, nj] == 1:
        #                     grid_graph.add_edge((i, j), (ni, nj))

        # Add edges between orthogonally adjacent cells ONLY
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for i in range(self.height):
            for j in range(self.width):
                if self.cell_map[i, j] == 1:
                    # Check only orthogonal neighbors
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.height and 
                            0 <= nj < self.width and 
                            self.cell_map[ni, nj] == 1):
                            grid_graph.add_edge((i, j), (ni, nj))
        
        
        return grid_graph
    
    def _is_complete_mega_cell(self, i, j):
        """Check if a 2x2 mega-cell starting at (i,j) is complete."""
        if i+1 >= self.height or j+1 >= self.width:
            return False
        
        return (self.cell_map[i, j] == 1 and 
                self.cell_map[i+1, j] == 1 and 
                self.cell_map[i, j+1] == 1 and 
                self.cell_map[i+1, j+1] == 1)
    
    def classify_cells(self):
        """
        Conduct cell classification (trunk, branch, and subsidiary cells)
        """
        self.trunk_cells = set()
        self.branch_cells = set()
        self.subsidiary_cells = set()
        
        # Identify complete mega-cells
        for i in range(self.height-1):
            for j in range(self.width-1):
                if self._is_complete_mega_cell(i, j):
                    self.trunk_cells.add((i, j))
                    self.trunk_cells.add((i+1, j))
                    self.trunk_cells.add((i, j+1))
                    self.trunk_cells.add((i+1, j+1))
        
        # Identify subsidiary cells (cells with only one neighbor)
        for node in self.grid_graph.nodes():
            if len(list(self.grid_graph.neighbors(node))) <= 1:
                self.subsidiary_cells.add(node)
        
        # Remaining cells are branch cells
        for i in range(self.height):
            for j in range(self.width):
                if self.cell_map[i, j] == 1:
                    cell = (i, j)
                    if cell not in self.trunk_cells and cell not in self.subsidiary_cells:
                        self.branch_cells.add(cell)
    
    def compute_path_points(self):
        """Compute path points for each cell."""
        for i in range(self.height):
            for j in range(self.width):
                if self.cell_map[i, j] == 1:
                    # Always use the exact center of the cell as the path point
                    # Robot always passes through the center of each cell
                    self.path_points[(i, j)] = (i + 0.5, j + 0.5)
    
    def find_hamiltonian_structures_in_branch(self):
        """Find Hamiltonian structures in the branch graph."""
        branch_graph = self.grid_graph.subgraph(self.branch_cells)
        hamiltonian_structures = []
        
        # 1. Search for cycles
        cycles = self._find_cycles(branch_graph)
        for cycle in cycles:
            if set(cycle).issubset(self.branch_cells):
                hamiltonian_structures.append(cycle)
        
        # 2. Search for 2xn grid graphs
        grid_patterns = self._find_2xn_patterns(branch_graph)
        for pattern in grid_patterns:
            hamiltonian_structures.append(pattern)
        
        # 3. Search for vertex pairs
        remaining_cells = self.branch_cells - set().union(*[set(h) for h in hamiltonian_structures]) if hamiltonian_structures else self.branch_cells
        pairs = self._find_vertex_pairs(branch_graph.subgraph(remaining_cells))
        for pair in pairs:
            hamiltonian_structures.append(pair)
        
        return hamiltonian_structures
    
    def _find_cycles(self, graph):
        """
        Find cycles in the graph.
        "In cases where a Hamiltonian cycle is not possible, we
        resort to the most basic Hamiltonian path structure, which
        comprises pairs of adjacent vertices."
        """
        try:
            cycles = list(nx.cycle_basis(graph))
            # Sort by size (prefer smaller cycles)
            cycles.sort(key=len)
            return cycles
        except nx.NetworkXNoCycle:
            # Handle case where there are no cycles
            return []
    
    def _find_2xn_patterns(self, graph):
        """Find 2xn grid patterns in the graph."""
        patterns = []
        visited = set()
        
        for node in graph.nodes():
            if node in visited:
                continue
            
            # Try to grow a 2xn pattern from this node
            i, j = node
            pattern = [(i, j)]
            visited.add(node)
            
            # Try to extend horizontally
            current_j = j + 1
            while (i, current_j) in graph.nodes() and (i, current_j) not in visited:
                # Check for orthogonal movement
                if self._is_orthogonal_neighbor((i, current_j-1), (i, current_j)):
                    pattern.append((i, current_j))
                    visited.add((i, current_j))
                    current_j += 1
                else:
                    break
            
            # Check if we can form a 2xn pattern
            if len(pattern) >= 2:
                # Check second row
                second_row = []
                for _, j_val in pattern:
                    if (i+1, j_val) in graph.nodes() and (i+1, j_val) not in visited:
                        # Ensure orthogonal connection
                        if self._is_orthogonal_neighbor((i, j_val), (i+1, j_val)):
                            second_row.append((i+1, j_val))
                            visited.add((i+1, j_val))
                        else:
                            break
                    else:
                        # Cannot form a complete 2xn pattern
                        break
                
                if len(second_row) == len(pattern):
                    patterns.append(pattern + second_row)
                else:
                    # Remove nodes from visited if pattern not complete
                    for node in second_row:
                        visited.remove(node)
        
        return patterns
    
    def _find_vertex_pairs(self, graph):
        """Find adjacent vertex pairs in the graph."""
        pairs = []
        visited = set()
        
        for node in graph.nodes():
            if node in visited:
                continue
            
            visited.add(node)
            # Find unvisited neighbors
            for neighbor in graph.neighbors(node):
                if neighbor not in visited and self._is_orthogonal_neighbor(node, neighbor):
                    pairs.append([node, neighbor])
                    visited.add(neighbor)
                    break
        
        return pairs
    
    def create_trunk_hamiltonian_cycle(self):
        """Create a Hamiltonian cycle in the trunk graph using STC algorithm."""
        if not self.trunk_cells:
            return []
        
        # Create a graph of mega-cells
        mega_cell_graph = nx.Graph()
        mega_cells = set()
        
        for i in range(0, self.height-1, 2):
            for j in range(0, self.width-1, 2):
                if self._is_complete_mega_cell(i, j):
                    mega_cells.add((i//2, j//2))
                    mega_cell_graph.add_node((i//2, j//2))
        
        # Add edges between adjacent mega-cells
        for mc1 in mega_cells:
            for mc2 in mega_cells:
                i1, j1 = mc1
                i2, j2 = mc2
                if (abs(i1-i2) == 1 and j1 == j2) or (abs(j1-j2) == 1 and i1 == i2):  # Only orthogonal connections
                    mega_cell_graph.add_edge(mc1, mc2)
        
        # Create minimum spanning tree (MST)
        if not nx.is_connected(mega_cell_graph):
            if mega_cell_graph.nodes():  # Check if the graph has any nodes
                largest_cc = max(nx.connected_components(mega_cell_graph), key=len)
                mega_cell_graph = mega_cell_graph.subgraph(largest_cc)
            else:
                return []  # Return empty cycle if no mega cells
        
        if not mega_cell_graph.edges():
            # If there are no edges in the graph (single node), return that node's cells
            if mega_cell_graph.nodes():
                i, j = list(mega_cell_graph.nodes())[0]
                cell_i, cell_j = i*2, j*2
                return [(cell_i, cell_j), (cell_i+1, cell_j), 
                        (cell_i+1, cell_j+1), (cell_i, cell_j+1)]
            return []
        
        mst = nx.minimum_spanning_tree(mega_cell_graph)
        
        # Generate Hamiltonian cycle by navigating around the MST
        hamiltonian_cycle = self._navigate_around_mst(mst)
        
        # Convert mega-cell cycle to cell coordinates
        cell_cycle = []
        for i, j in hamiltonian_cycle:
            # Each mega-cell corresponds to 4 cells
            cell_i, cell_j = i*2, j*2
            cell_cycle.extend([(cell_i, cell_j), (cell_i+1, cell_j), 
                               (cell_i+1, cell_j+1), (cell_i, cell_j+1)])
        
        return cell_cycle

    def extend_hamiltonian_cycle(self, trunk_cycle, branch_structures):
        """Extend the trunk Hamiltonian cycle with branch structures."""
        if not trunk_cycle:
            # If trunk cycle is empty, just concatenate all branch structures
            extended_cycle = []
            for structure in branch_structures:
                extended_cycle.extend(structure)
            return extended_cycle
        
        extended_cycle = trunk_cycle.copy()
        
        # For each branch structure, find the best place to integrate it into the trunk cycle
        for structure in branch_structures:
            if not structure:  # Skip empty structures
                continue
                
            best_insertion_point = 0
            best_additional_distance = float('inf')
            best_connection = None
            
            # Try all possible insertion points in the extended cycle
            for i in range(len(extended_cycle)):
                cell1 = extended_cycle[i]
                cell2 = extended_cycle[(i+1) % len(extended_cycle)]

                # Skip if cells not in grid map coverage area
                if (self.cell_map[cell1[0], cell1[1]] == 0 or 
                    self.cell_map[cell2[0], cell2[1]] == 0):
                    continue
                
                # Check potential connections from cell1 to structure
                for struct_idx, struct_cell in enumerate(structure):
                    if struct_cell in self.grid_graph.neighbors(cell1):
                        # Connect cell1 to the start of the structure
                        # Then connect the end of the structure to cell2
                        end_cell = structure[(struct_idx - 1) % len(structure)]
                        
                        if end_cell in self.grid_graph.neighbors(cell2):
                            # Calculate additional distance with this insertion
                            original_dist = self._distance(cell1, cell2)
                            
                            # Rearrange structure to start with struct_cell and check distance
                            arranged_structure = structure[struct_idx:] + structure[:struct_idx]
                            new_dist = self._distance(cell1, arranged_structure[0])
                            for j in range(len(arranged_structure) - 1):
                                new_dist += self._distance(arranged_structure[j], arranged_structure[j+1])
                            new_dist += self._distance(arranged_structure[-1], cell2)
                            
                            additional_dist = new_dist - original_dist
                            
                            if additional_dist < best_additional_distance:
                                best_additional_distance = additional_dist
                                best_insertion_point = i
                                best_connection = (struct_idx, arranged_structure)
            
            # Insert the best found structure into the cycle
            if best_connection is not None:
                _, arranged_structure = best_connection
                extended_cycle = extended_cycle[:best_insertion_point+1] + arranged_structure + extended_cycle[best_insertion_point+1:]
        
        return extended_cycle
    
    def _navigate_around_mst(self, mst):
        """Create a proper Hamiltonian cycle by navigating around the MST.
        This is the core of the STC algorithm approach."""
        if not mst.nodes():
            return []
        
        # The approach is to navigate around the MST, keeping it to our right
        # Start with some node that has a degree of 1 (a leaf in the MST)
        start_node = None
        for node in mst.nodes():
            if mst.degree(node) == 1:
                start_node = node
                break
        
        if start_node is None:
            # If no leaf node found (could be a cycle), pick any node
            start_node = list(mst.nodes())[0]
        
        # Get the initial direction
        if mst.degree(start_node) == 0:  # Isolated node
            return [start_node]
            
        # Find the neighbor to start with
        next_node = list(mst.neighbors(start_node))[0]
        
        # Create the walk
        cycle = [start_node]
        visited_edges = set()
        edge = tuple(sorted([start_node, next_node]))
        visited_edges.add(edge)
        
        # Continue the walk until we return to the start node
        current = next_node
        
        while True:
            cycle.append(current)
            
            # Get all neighbors of the current node sorted by their position
            # This ensures we always move in the same direction around the MST
            neighbors = sorted(list(mst.neighbors(current)))
            
            # Find an unvisited edge
            next_node = None
            for neighbor in neighbors:
                edge = tuple(sorted([current, neighbor]))
                if edge not in visited_edges:
                    next_node = neighbor
                    visited_edges.add(edge)
                    break
            
            if next_node is None:
                # If all edges from current node are visited, backtrack
                for neighbor in neighbors:
                    # If we can get back to the start node, close the cycle
                    if neighbor == start_node and len(cycle) > 2:
                        return cycle + [start_node]
                
                # Backtrack to find a node with unvisited edges
                for i in range(len(cycle)-2, -1, -1):
                    backtrack_node = cycle[i]
                    for neighbor in mst.neighbors(backtrack_node):
                        edge = tuple(sorted([backtrack_node, neighbor]))
                        if edge not in visited_edges:
                            # Trim the cycle and continue from backtrack_node
                            cycle = cycle[:i+1]
                            current = backtrack_node
                            next_node = neighbor
                            visited_edges.add(edge)
                            break
                    if next_node is not None:
                        break
                
                # If no unvisited edges found, and we're back at start_node, we're done
                if next_node is None:
                    if current == start_node:
                        return cycle
                    else:
                        # We need to get back to the start node
                        try:
                            path = nx.shortest_path(mst, current, start_node)
                            for node in path[1:]:
                                cycle.append(node)
                            return cycle
                        except nx.NetworkXNoPath:
                            # If no path exists, just return what we have
                            return cycle
            else:
                current = next_node
                
                # If we've come back to the start node, we're done
                if current == start_node:
                    return cycle
    
    def add_subsidiary_cells(self, cycle):
        """Add subsidiary cells to the cycle with minimal path increase."""
        if not cycle:
            # If the cycle is empty, just return all subsidiary cells in any order
            return list(self.subsidiary_cells)
        
        complete_path = cycle.copy()
        remaining_subsidiaries = self.subsidiary_cells.copy()
        
        # Iteratively add subsidiary cells to minimize additional path length
        while remaining_subsidiaries:
            best_insertion = None
            best_insertion_point = 0
            min_additional_distance = float('inf')
            
            # For each remaining subsidiary cell - use list() to create a copy for iteration
            for cell in list(remaining_subsidiaries):
                # Skip if the cell is not in path_points (which should not happen but as a safeguard)
                if cell not in self.path_points:
                    remaining_subsidiaries.remove(cell)
                    continue
                    
                # Try inserting after each point in the current path
                for i in range(len(complete_path)):
                    prev_cell = complete_path[i]
                    next_cell = complete_path[(i+1) % len(complete_path)]
                    
                    # Skip if either cell is not in path_points
                    if prev_cell not in self.path_points or next_cell not in self.path_points:
                        continue
                    
                    # Check if this subsidiary cell can be connected here
                    if cell in self.grid_graph.neighbors(prev_cell) or cell in self.grid_graph.neighbors(next_cell):
                        # Calculate additional distance if we insert this subsidiary cell
                        original_dist = self._distance(prev_cell, next_cell)
                        new_dist = self._distance(prev_cell, cell) + self._distance(cell, next_cell)
                        additional_dist = new_dist - original_dist
                        
                        if additional_dist < min_additional_distance:
                            min_additional_distance = additional_dist
                            best_insertion_point = (i + 1) % len(complete_path)
                            best_insertion = cell
            
            # If a good insertion, add it to the path
            if best_insertion is not None:
                complete_path.insert(best_insertion_point, best_insertion)
                remaining_subsidiaries.remove(best_insertion)
            else:
                # If we can't connect any more subsidiaries, break
                break
        
        # If there are still remaining subsidiaries that couldn't be connected directly
        # Find the shortest possible detour to reach them
        if remaining_subsidiaries:
            # Create a copy of remaining_subsidiaries to avoid modification during iteration
            cells_to_process = list(remaining_subsidiaries)
            # Find shortest paths to all remaining subsidiaries
            for cell in cells_to_process:
                if cell not in self.path_points:  # Skip if not in path_points
                    remaining_subsidiaries.remove(cell)
                    continue
                    
                closest_point = None
                min_distance = float('inf')
                best_position = 0
                best_path = None
                
                # Find the closest point in the current path
                for i, path_point in enumerate(complete_path):
                    if path_point not in self.path_points:  # Skip if not in path_points
                        continue
                        
                    try:
                        # Find shortest path using A* algorithm
                        # shortest_path = nx.astar_path(self.grid_graph, path_point, cell)
                        shortest_path = nx.astar_path(self.grid_graph, path_point, cell, 
                            heuristic=lambda a, b: self._distance(a, b))
                        
                        valid_path = True
                        for path_cell in shortest_path:
                            if self.cell_map[path_cell[0], path_cell[1]] == 0:
                                valid_path = False
                                break
                        dist = len(shortest_path) - 1  # Length of path minus 1 = number of edges
                        
                        if dist < min_distance:
                            min_distance = dist
                            closest_point = path_point
                            best_position = i
                            best_path = shortest_path
                    except nx.NetworkXNoPath:
                        continue
                
                if closest_point is not None and best_path is not None:
                    # Get the shortest path and insert it (excluding the start point)
                    for j, point in enumerate(best_path[1:]):
                        complete_path.insert(best_position + j + 1, point)
                    
                    # Mark cell as visited
                    if cell in remaining_subsidiaries:
                        remaining_subsidiaries.remove(cell)
        
        return complete_path
    
    def euclidean_distance(self, cell1, cell2):
        """Calculate Euclidean distance between path points of two cells."""
        # Check if both cells are in path_points
        if cell1 in self.path_points and cell2 in self.path_points:
            x1, y1 = self.path_points[cell1]
            x2, y2 = self.path_points[cell2]
            return np.sqrt((x2-x1)**2 + (y2-y1)**2)
        else:
            # If either cell is not in path_points, return a large distance
            return float('inf')
        
    def _distance(self, cell1, cell2):
        """
        Calculate Manhattan distance between two cells (no diagonal movement).
        """
        i1, j1 = cell1
        i2, j2 = cell2
        return abs(i1 - i2) + abs(j1 - j2)
    
    def _is_orthogonal_neighbor(self, cell1, cell2):
        """Check if two cells are orthogonal neighbors (no diagonal)."""
        i1, j1 = cell1
        i2, j2 = cell2
        manhattan_dist = abs(i1 - i2) + abs(j1 - j2)
        return manhattan_dist == 1  # Exactly 1 step in Manhattan distance = orthogonal neighbor
    
    def plan_coverage_path(self):
        """Plan the complete coverage path."""
        # Step 1: Do cell classification
        self.classify_cells()
        
        # Step 2: Compute path points
        self.compute_path_points()
        
        # Step 3: Find Hamiltonian cycle in trunk cells
        trunk_cycle = self.create_trunk_hamiltonian_cycle()
        
        # Step 4: Find Hamiltonian structures in branch cells
        branch_structures = self.find_hamiltonian_structures_in_branch()
        
        # Step 5: Extend the trunk cycle with branch structures
        extended_cycle = self.extend_hamiltonian_cycle(trunk_cycle, branch_structures)
        
        # Step 6: Add subsidiary cells
        complete_path = self.add_subsidiary_cells(extended_cycle)
        
        # Convert cell path to path points
        path_points = []
        for cell in complete_path:
            if cell in self.path_points:
                path_points.append(self.path_points[cell])
        
        return complete_path, path_points
    
    def visualize(self, path=None):
        """Visualize the region, cells, and coverage path."""
        plt.figure(figsize=(12, 10))
        
        # Create a colored visualization of the cell classifications
        visualization_map = np.zeros((self.height, self.width, 3))
        
        # Set colors according to the requested scheme
        # Dark purple for cells exempt from coverage
        dark_purple = np.array([0.3, 0.0, 0.5])  # RGB for dark purple
        yellow = np.array([1.0, 1.0, 0.0])  # RGB for yellow (trunk cells)
        green = np.array([0.0, 0.8, 0.0])   # RGB for green (branch cells)
        green_blue = np.array([0.0, 0.7, 0.7])  # RGB for green-blue (subsidiary cells)
        
        # Fill the visualization map with the appropriate colors
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.trunk_cells:
                    visualization_map[i, j] = yellow
                elif (i, j) in self.branch_cells:
                    visualization_map[i, j] = green
                elif (i, j) in self.subsidiary_cells:
                    visualization_map[i, j] = green_blue
                elif self.grid_map[i, j] == 0:  # Area to be covered but not classified
                    visualization_map[i, j] = np.array([0.9, 0.9, 0.9])
                else:  # grid_map[i, j] == 1, exempt from coverage
                    visualization_map[i, j] = dark_purple
        
        # Plot the colored map
        plt.imshow(visualization_map, interpolation='nearest', 
                  extent=[-0.5, self.width-0.5, self.height-0.5, -0.5])
        
        # Add grid for each cell with dotted black lines
        for i in range(self.height + 1):
            plt.axhline(y=i-0.5, color='black', linestyle=':', alpha=0.7, linewidth=0.8)
        for j in range(self.width + 1):
            plt.axvline(x=j-0.5, color='black', linestyle=':', alpha=0.7, linewidth=0.8)
        
        # Add highlighted grid for mega-cells (2x2)
        for i in range(0, self.height + 1, 2):
            plt.axhline(y=i-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
        for j in range(0, self.width + 1, 2):
            plt.axvline(x=j-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
        
        # Plot path if provided
        if path:
            # Extract path points that go through cell centers
            path_x = []
            path_y = []
            for cell in path:
                # Get exact center of the cell
                i, j = cell
                path_y.append(i)
                path_x.append(j)
            
            # Plot the path
            plt.plot(path_x, path_y, 'r-', linewidth=2)  # Red path for better visibility
            
            # Mark path nodes
            plt.plot(path_x, path_y, 'ro', markersize=3, alpha=0.5)  # Small red circles at cell centers
            
            # Mark start point
            plt.plot(path_x[0], path_y[0], 'ks', markersize=8)  # Start point as black square
        
        plt.title('Coverage Path Planning', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.xticks(range(0, self.width))
        plt.yticks(range(0, self.height))
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=yellow, label='Trunk Cells'),
            Patch(facecolor=green, label='Branch Cells'),
            Patch(facecolor=green_blue, label='Subsidiary Cells'),
            Patch(facecolor=dark_purple, label='Exempt from Coverage'),
            Patch(facecolor='red', label='Coverage Path')
        ]
        # plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()

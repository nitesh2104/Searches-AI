# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""
import traceback
from heapq import heappush, heappop

import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""
        self.key_count_dict = {}
        self.queue = []
        self.count = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        val = heappop(self.queue)
        del val[1]
        return val

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        for cost, priority, path_node in self.queue:
            if node == path_node:
                self.queue.remove([cost, priority, path_node])

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        self.count += 1
        new_node = [node[0], self.count] + list(node[1:])
        heappush(self.queue, new_node)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
        path = list(letter states)
    """
    if start == goal:
        return []
    frontier = []
    explored = set()
    frontier.append([start])
    explored.add(start)
    while True:
        if not frontier:
            return False
        path = frontier.pop(0)
        new_frontiers = sorted(graph[path[-1]].items(), key=lambda x: x[0])
        for new_frontier, weights in new_frontiers:
            if new_frontier not in explored:
                explored.add(new_frontier)
                if new_frontier == goal:
                    return path + [new_frontier]
                frontier.append(path + [new_frontier])


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = PriorityQueue()
    explored = set()
    frontier.append([0, [start]])
    while True:
        if not frontier.size():
            return False
        curr_cost, path = frontier.pop()
        if goal == path[-1]:
            return path
        if path[-1] not in explored:
            explored.add(path[-1])
            for new_frontier, weights in sorted(graph[path[-1]].items(), key=lambda x: x[1]['weight']):
                if new_frontier not in explored:
                    frontier.append((curr_cost + weights["weight"], path + [new_frontier]))


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """
    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    pos_a = graph.nodes[v]['pos']
    pos_b = graph.nodes[goal]['pos']
    return ((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2) ** 0.5


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = PriorityQueue()
    explored = set()
    frontier.append([heuristic(graph, start, goal), 0, [start]])
    while True:
        if not frontier.size():
            return False
        curr_cost_goal, curr_path_cost, path = frontier.pop()
        if goal == path[-1]:
            return path
        if path[-1] not in explored:
            explored.add(path[-1])
            for new_frontier, weight in sorted(graph[path[-1]].items(), key=lambda x: x[1]["weight"]):
                if new_frontier not in explored:
                    frontier.append(
                        [curr_path_cost + weight['weight'] + euclidean_dist_heuristic(graph, new_frontier, goal),
                         curr_path_cost + weight['weight'], path + [new_frontier]])


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()
    final_path = PriorityQueue()
    explored_start = set()
    explored_goal = set()
    frontier_start.append([0, [start]])
    frontier_goal.append([0, [goal]])
    while True:
        if not frontier_start.size() or not frontier_goal.size():
            return final_path.queue[0][-1] if final_path.size() else False

        if final_path.size() and frontier_start.queue[0][0] + frontier_goal.queue[0][0] > final_path.queue[0][0]:
            return final_path.queue[0][-1]

        # start
        curr_cost, path = frontier_start.pop()
        if goal == path[-1]:
            return path
        for goal_frontier in frontier_goal.queue:
            path_cost, _, goal_path = goal_frontier
            if path[-1] in goal_path:
                final_path.append([curr_cost + path_cost, path + goal_path[::-1][1:]])
        if path[-1] not in explored_start:
            explored_start.add(path[-1])
            for new_frontier, weights in sorted(graph[path[-1]].items(), key=lambda x: x[1]['weight']):
                if new_frontier not in explored_start:
                    frontier_start.append((curr_cost + weights["weight"], path + [new_frontier]))

        # goal
        curr_cost, path = frontier_goal.pop()
        if start == path[-1]:
            return path[::-1]
        for start_frontier in frontier_start.queue:
            path_cost, _, start_path = start_frontier
            if path[-1] in start_path:
                final_path.append([curr_cost + path_cost, start_path + path[::-1][1:]])
        if path[-1] not in explored_goal:
            explored_goal.add(path[-1])
            for new_frontier, weights in sorted(graph[path[-1]].items(), key=lambda x: x[1]['weight']):
                if new_frontier not in explored_start:
                    frontier_goal.append((curr_cost + weights["weight"], path + [new_frontier]))


def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier_start, frontier_goal, final_path = PriorityQueue(), PriorityQueue(), PriorityQueue()
    explored_start, explored_goal = set(), set()
    frontier_start.append([heuristic(graph, start, goal), 0, [start]])
    frontier_goal.append([heuristic(graph, start, goal), 0, [goal]])
    while True:
        if not frontier_start.size() or not frontier_goal.size():
            return final_path.queue[0][-1] if final_path.size() else False

        if final_path.size():
            if sorted(frontier_start.queue, key=lambda x: x[2])[0][2] + \
                    sorted(frontier_goal.queue, key=lambda x: x[2])[0][2] > final_path.queue[0][0]:
                return final_path.queue[0][-1]

        # start
        curr_cost_goal, curr_path_cost, path = frontier_start.pop()
        if goal == path[-1]:
            return path
        for goal_frontier in frontier_goal.queue:
            path_cost, _, curr_cost, goal_path = goal_frontier
            if path[-1] in goal_path:
                final_path.append([curr_path_cost + curr_cost, path + goal_path[::-1][1:]])
        if path[-1] not in explored_start:
            explored_start.add(path[-1])
            for new_frontier, weights in sorted(graph[path[-1]].items(), key=lambda x: x[1]['weight']):
                if new_frontier not in explored_start:
                    frontier_start.append(
                        [curr_path_cost + weights['weight'] + euclidean_dist_heuristic(graph, new_frontier, goal),
                         curr_path_cost + weights['weight'], path + [new_frontier]])

        # goal
        curr_cost_goal, curr_path_cost, path = frontier_goal.pop()
        if start == path[-1]:
            return path[::-1]
        for start_frontier in frontier_start.queue:
            path_cost, _, curr_cost, start_path = start_frontier
            if path[-1] in start_path:
                final_path.append([curr_path_cost + curr_cost, start_path + path[::-1][1:]])
        if path[-1] not in explored_goal:
            explored_goal.add(path[-1])
            for new_frontier, weights in sorted(graph[path[-1]].items(), key=lambda x: x[1]['weight']):
                if new_frontier not in explored_start:
                    frontier_goal.append(
                        [curr_path_cost + weights['weight'] + euclidean_dist_heuristic(graph, new_frontier, goal),
                         curr_path_cost + weights['weight'], path + [new_frontier]])


def last_node(queue):
    return queue[-1]

def merged_path(path1, path2, path3, goal):
    if all([x in path1[1][1] for x in goal]): return path1[1][1]
    if all([x in path2[1][1] for x in goal]): return path2[1][1]
    if all([x in path3[1][1] for x in goal]): return path3[1][1]

    merged_paths = sorted([path1] + [path2] + [path3], key=lambda x: x[1][0])
    path1 = last_node(merged_paths[0][1])
    path2 = last_node(merged_paths[1][1])

    if path1 == path2: return path1
    elif all([x in path2 for x in path1]): return path2
    elif all([x in path1 for x in path2]): return path1
    else:
        if path1[0] == path2[0]: return path1[::-1][:-1] + path2
        elif path1[-1] == path2[0]: return path1[:-1] + path2
        elif path1[0] == path2[-1]: return path2[:-1] + path1
        elif path1[-1] == path2[-1]: return path1[:-1] + path2[::-1]
        else:
            print("ERRORRRRR")
            return False


def tridirectional_search(graph, goal):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goal[0] == goal[1] == goal[2]:
        return []

    frontier_0, frontier_1, frontier_2, final_path = [PriorityQueue() for _ in range(4)]
    explored_0, explored_1, explored_2 = [set() for _ in range(3)]
    g0g1_found, g1g2_found, g0g2_found = [False, []], [False, []], [False,[]]
    frontier_0.append([0, [goal[0]]])
    frontier_1.append([0, [goal[1]]])
    frontier_2.append([0, [goal[2]]])

    while True:
        if not frontier_0.size() or not frontier_1.size() or not frontier_2.size():
            return merged_path(g0g1_found, g1g2_found, g0g2_found, goal)

        if final_path.count:
            for final_path_cost, _, fpath in final_path.queue:
                if goal[0] in fpath and goal[1] in fpath and not g0g1_found:
                    if frontier_0.queue[0][0] + frontier_1.queue[0][0] > final_path_cost:
                        g0g1_found = [True, [final_path_cost, final_path]]
                if goal[1] in fpath and goal[2] in fpath and not g1g2_found:
                    if frontier_1.queue[0][0] + frontier_2.queue[0][0] > final_path_cost:
                        g1g2_found = [True, [final_path_cost, final_path]]
                if goal[0] in fpath and goal[2] in fpath and not g0g2_found:
                    if frontier_0.queue[0][0] + frontier_2.queue[0][0] > final_path_cost:
                        g0g2_found = [True, [final_path_cost, final_path]]

        if g0g1_found[0] and g1g2_found[0] and g0g2_found[0]:
            return merged_path(g0g1_found, g1g2_found, g0g2_found, goal)

        if frontier_0.size():
            curr_path_cost, current_path = frontier_0.pop()
            if last_node(current_path) == goal[1]:
                final_path.append([curr_path_cost, current_path])
                if not g0g1_found[0] or curr_path_cost < g0g1_found[1][0]:
                    g0g1_found = [True, [curr_path_cost, current_path]]
            if last_node(current_path) == goal[2]:
                final_path.append([curr_path_cost, current_path])
                if not g0g2_found[0] or curr_path_cost < g0g2_found[1][0]:
                    g0g2_found = [True, [curr_path_cost, current_path]]
            if not g0g1_found[0]:
                for goal_path_cost, _, goal_path in frontier_1.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if not g0g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_2.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if last_node(current_path) not in explored_0:
                explored_0.add(last_node(current_path))
                for new_frontier, frontier_weight in sorted(graph[last_node(current_path)].items(), key=lambda x: x[1]['weight']):
                    if new_frontier not in explored_0:
                        frontier_0.append([curr_path_cost + frontier_weight["weight"], current_path + [new_frontier]])

        if frontier_1.size():
            curr_path_cost, current_path = frontier_1.pop()
            if last_node(current_path) == goal[0]:
                final_path.append([curr_path_cost, current_path])
                if not g0g1_found[0] or curr_path_cost < g0g1_found[1][0]:
                    g0g1_found = [True, [curr_path_cost, current_path]]
            if last_node(current_path) == goal[2]:
                final_path.append([curr_path_cost, current_path])
                if not g1g2_found[0] or curr_path_cost < g1g2_found[1][0]:
                    g1g2_found = [True, [curr_path_cost, current_path]]
            if not g0g1_found[0]:
                for goal_path_cost, _, goal_path in frontier_0.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if not g1g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_2.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if last_node(current_path) not in explored_1:
                explored_1.add(last_node(current_path))
                for new_frontier, frontier_weight in sorted(graph[last_node(current_path)].items(), key=lambda x: x[1]['weight']):
                    if new_frontier not in explored_1:
                        frontier_1.append([curr_path_cost + frontier_weight["weight"], current_path + [new_frontier]])

        if frontier_2.size():
            curr_path_cost, current_path = frontier_2.pop()
            if last_node(current_path) == goal[0]:
                final_path.append([curr_path_cost, current_path])
                if not g0g2_found[0] or curr_path_cost < g0g2_found[1][0]:
                    g0g2_found = [True, [curr_path_cost, current_path]]
            if last_node(current_path) == goal[1]:
                final_path.append([curr_path_cost, current_path])
                if not g1g2_found[0] or curr_path_cost < g1g2_found[1][0]:
                    g1g2_found = [True, [curr_path_cost, current_path]]
            if not g0g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_0.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if not g1g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_1.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if last_node(current_path) not in explored_2:
                explored_2.add(last_node(current_path))
                for new_frontier, frontier_weight in sorted(graph[last_node(current_path)].items(), key=lambda x: x[1]['weight']):
                    if new_frontier not in explored_2:
                        frontier_2.append([curr_path_cost + frontier_weight["weight"], current_path + [new_frontier]])



def tridirectional_upgraded(graph, goal, heuristic=euclidean_dist_heuristic):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    frontier_0, frontier_1, frontier_2, final_path = [PriorityQueue() for _ in range(4)]
    explored_0, explored_1, explored_2 = [set() for _ in range(3)]
    g0g1_found, g1g2_found, g0g2_found = [False, []], [False, []], [False, []]
    frontier_0.append([0, [goal[0]]])
    frontier_1.append([0, [goal[1]]])
    frontier_2.append([0, [goal[2]]])

    while True:
        if not frontier_0.size() or not frontier_1.size() or not frontier_2.size():
            return last_node(final_path.queue[0]) if final_path.size() else False

        if final_path.count:
            for final_path_cost, _, fpath in final_path.queue:
                if goal[0] in fpath and goal[1] in fpath and not g0g1_found:
                    if frontier_0.queue[0][0] + frontier_1.queue[0][0] > final_path_cost:
                        g0g1_found = [True, [final_path_cost, final_path]]
                if goal[1] in fpath and goal[2] in fpath and not g1g2_found:
                    if frontier_1.queue[0][0] + frontier_2.queue[0][0] > final_path_cost:
                        g1g2_found = [True, [final_path_cost, final_path]]
                if goal[0] in fpath and goal[2] in fpath and not g0g2_found:
                    if frontier_0.queue[0][0] + frontier_2.queue[0][0] > final_path_cost:
                        g0g2_found = [True, [final_path_cost, final_path]]

        if g0g1_found[0] and g1g2_found[0] and g0g2_found[0]:
            return merged_path(g0g1_found, g1g2_found, g0g2_found, goal)

        if frontier_0.size():
            curr_path_cost, current_path = frontier_0.pop()
            if last_node(current_path) == goal[1]:
                final_path.append([curr_path_cost, current_path])
                if not g0g1_found[0] or curr_path_cost < g0g1_found[1][0]:
                    g0g1_found = [True, [curr_path_cost, current_path]]
            if last_node(current_path) == goal[2]:
                final_path.append([curr_path_cost, current_path])
                if not g0g2_found[0] or curr_path_cost < g0g2_found[1][0]:
                    g0g2_found = [True, [curr_path_cost, current_path]]
            if not g0g1_found[0]:
                for goal_path_cost, _, goal_path in frontier_1.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if not g0g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_2.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if last_node(current_path) not in explored_0:
                explored_0.add(last_node(current_path))
                for new_frontier, frontier_weight in sorted(graph[last_node(current_path)].items(), key=lambda x: x[1]['weight']):
                    if new_frontier not in explored_0:
                        frontier_0.append([curr_path_cost + frontier_weight["weight"], current_path + [new_frontier]])

        if frontier_1.size():
            curr_path_cost, current_path = frontier_1.pop()
            if last_node(current_path) == goal[0]:
                final_path.append([curr_path_cost, current_path])
                if not g0g1_found[0] or curr_path_cost < g0g1_found[1][0]:
                    g0g1_found = [True, [curr_path_cost, current_path]]
            if last_node(current_path) == goal[2]:
                final_path.append([curr_path_cost, current_path])
                if not g1g2_found[0] or curr_path_cost < g1g2_found[1][0]:
                    g1g2_found = [True, [curr_path_cost, current_path]]
            if not g0g1_found[0]:
                for goal_path_cost, _, goal_path in frontier_0.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if not g1g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_2.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if last_node(current_path) not in explored_1:
                explored_1.add(last_node(current_path))
                for new_frontier, frontier_weight in sorted(graph[last_node(current_path)].items(), key=lambda x: x[1]['weight']):
                    if new_frontier not in explored_1:
                        frontier_1.append([curr_path_cost + frontier_weight["weight"], current_path + [new_frontier]])

        if frontier_2.size():
            curr_path_cost, current_path = frontier_2.pop()
            if last_node(current_path) == goal[0]:
                final_path.append([curr_path_cost, current_path])
                if not g0g2_found[0] or curr_path_cost < g0g2_found[1][0]:
                    g0g2_found = [True, [curr_path_cost, current_path]]
            if last_node(current_path) == goal[1]:
                final_path.append([curr_path_cost, current_path])
                if not g1g2_found[0] or curr_path_cost < g1g2_found[1][0]:
                    g1g2_found = [True, [curr_path_cost, current_path]]
            if not g0g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_0.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if not g1g2_found[0]:
                for goal_path_cost, _, goal_path in frontier_1.queue:
                    if last_node(current_path) in goal_path:
                        final_path.append([curr_path_cost + goal_path_cost, current_path + goal_path[::-1][1:]])
            if last_node(current_path) not in explored_2:
                explored_2.add(last_node(current_path))
                for new_frontier, frontier_weight in sorted(graph[last_node(current_path)].items(), key=lambda x: x[1]['weight']):
                    if new_frontier not in explored_2:
                        frontier_2.append([curr_path_cost + frontier_weight["weight"], current_path + [new_frontier]])


def return_your_name():
    """Return your name from this function"""
    name = "Nitesh Arora"
    return name


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # For now just adding a Bi-UCS, will add more changes later on
    if start == goal: return []
    frontier_start, frontier_goal, final_path = PriorityQueue(), PriorityQueue(), PriorityQueue()
    explored_start, explored_goal = set(), set()
    frontier_start.append([0, [start]])
    frontier_goal.append([0, [goal]])
    while True:
        if not frontier_start.size() or not frontier_goal.size():
            return final_path.queue[0][-1] if final_path.size() else False

        if final_path.size() and frontier_start.queue[0][0] + frontier_goal.queue[0][0] > final_path.queue[0][0]:
            return final_path.queue[0][-1]

        # start
        curr_cost, path = frontier_start.pop()
        if goal == path[-1]:
            return path
        for goal_frontier in frontier_goal.queue:
            path_cost, _, goal_path = goal_frontier
            if path[-1] in goal_path:
                final_path.append([curr_cost + path_cost, path + goal_path[::-1][1:]])
        if path[-1] not in explored_start:
            explored_start.add(path[-1])
            for new_frontier, weights in sorted(graph[path[-1]].items(), key=lambda x: x[1]['weight']):
                if new_frontier not in explored_start:
                    frontier_start.append((curr_cost + weights["weight"], path + [new_frontier]))

        # goal
        curr_cost, path = frontier_goal.pop()
        if start == path[-1]:
            return path[::-1]
        for start_frontier in frontier_start.queue:
            path_cost, _, start_path = start_frontier
            if path[-1] in start_path:
                final_path.append([curr_cost + path_cost, start_path + path[::-1][1:]])
        if path[-1] not in explored_goal:
            explored_goal.add(path[-1])
            for new_frontier, weights in sorted(graph[path[-1]].items(), key=lambda x: x[1]['weight']):
                if new_frontier not in explored_start:
                    frontier_goal.append((curr_cost + weights["weight"], path + [new_frontier]))


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    constOutFront = 2 * 6371  # Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0] - vLatLong[0]) / 2)) ** 2  # First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0]) * math.cos(goalLatLong[0]) * (
            (math.sin((goalLatLong[1] - vLatLong[1]) / 2)) ** 2)  # Second term
    return constOutFront * math.asin(math.sqrt(term1InSqrt + term2InSqrt))  # Straight application of formula
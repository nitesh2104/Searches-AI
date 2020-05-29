# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

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

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        return heappop(self.queue)

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        raise NotImplementedError

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
        if not node[0] in self.key_count_dict.keys():
            heappush(self.queue, (node[0], 0, node[1]))
            self.key_count_dict[node[0]] = 0
        else:
            self.key_count_dict[node[0]] += 1
            heappush(self.queue, (node[0], self.key_count_dict[node[0]], node[1]))

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

    frontier.append(start)

    explored.add(start)

    while True:
        if not frontier:
            return False

        path = frontier.pop(0)

        node_to_expand = path[-1]

        new_frontiers = graph[node_to_expand]

        for new_frontier in new_frontiers:

            if new_frontier not in explored:

                explored.add(new_frontier)
                if type(path) == list:
                    path.append(new_frontier)
                    final_path = path
                else:
                    final_path = [path, new_frontier]

                if new_frontier == goal:
                    return final_path

                frontier.append(final_path)


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
    final_path_list = []
    path_cost = 0
    heappush(frontier.queue, (path_cost, start))

    while frontier.size() != 0:
        if frontier.size() == 0:
            return False

        path = heappop(frontier.queue)

        if goal in path[1]:
            return path[1]

        node_to_expand = path[1][-1]

        explored.add(node_to_expand)

        new_frontiers = sorted(graph[node_to_expand].items(), key=lambda x: x[1]["weight"])

        for new_frontier, frontier_weight in new_frontiers:

            if new_frontier not in explored:

                path_cost = path[0] + frontier_weight["weight"]

                if type(path[1]) == str:
                    fpath = (path[1],) + (new_frontier,)
                else:
                    fpath = path[1] + (new_frontier,)
                final_path = (path_cost, (fpath))

                if new_frontier == goal:
                    if not final_path_list:
                        final_path_list = final_path
                    else:
                        if path_cost < final_path_list[0]:
                            final_path_list = final_path
                    continue

                heappush(frontier.queue, final_path)

    return final_path_list[1] if final_path_list else False


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
    path_cost = 0
    distance_from_goal = heuristic(graph, start, goal)
    final_path_list = []
    heappush(frontier.queue, (path_cost + distance_from_goal, start))

    while frontier.size() != 0:
        if frontier.size() == 0:
            return False

        path = heappop(frontier.queue)

        node_to_expand = path[1][-1]
        explored.add(node_to_expand)

        new_frontiers = graph[node_to_expand].items()

        for new_frontier, weight in new_frontiers:
            if new_frontier not in explored:
                path_cost = weight['weight'] + euclidean_dist_heuristic(graph, new_frontier, goal)

                if type(path[1]) == str:
                    fpath = (path[1],) + (new_frontier,)
                else:
                    fpath = path[1] + (new_frontier,)

                final_path = (path_cost, (fpath))

                if new_frontier == goal:
                    if not final_path_list:
                        final_path_list = final_path
                    else:
                        if path_cost < final_path_list[0]:
                            final_path_list = final_path
                    continue
                heappush(frontier.queue, final_path)

    return False if not final_path_list else final_path_list[1]


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
    explored_start = set()
    path_cost_start = 0

    frontier_goal = PriorityQueue()
    explored_goal = set()
    path_cost_goal = 0

    final_path_list = []

    heappush(frontier_start.queue, (path_cost_start, start))
    heappush(frontier_goal.queue, (path_cost_goal, goal))

    while frontier_start.size() != 0 and frontier_goal.size() != 0:
        if frontier_start.size() == 0:
            return False

        path = heappop(frontier_start.queue)

        if goal in path[1]:
            return path[1]

        node_to_expand = path[1][-1]

        explored_start.add(node_to_expand)

        new_frontiers = sorted(graph[node_to_expand].items(), key=lambda x: x[1]["weight"])

        for new_frontier, frontier_weight in new_frontiers:

            if new_frontier not in explored_start:

                path_cost_start = path[0] + frontier_weight["weight"]

                if type(path[1]) == str:
                    fpath = (path[1],) + (new_frontier,)
                else:
                    fpath = path[1] + (new_frontier,)
                final_path = (path_cost_start, (fpath))

                if new_frontier == goal:
                    if not final_path_list:
                        final_path_list = final_path
                    else:
                        if path_cost_start < final_path_list[0]:
                            final_path_list = final_path
                    continue

                heappush(frontier_start.queue, final_path)

    return final_path_list[1] if final_path_list else False


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

    # TODO: finish this function!
    raise NotImplementedError


def tridirectional_search(graph, goals):
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
    # TODO: finish this function
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
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
    # TODO: finish this function
    raise NotImplementedError


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

    # TODO: finish this function!
    raise NotImplementedError


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

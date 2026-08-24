# Graph Search: Bidirectional and Tridirectional Pathfinding

A from-scratch implementation of classical graph search algorithms, building up to **bidirectional A\*** and **tridirectional search** — finding a minimum-cost path that connects three nodes, not just two.

Benchmarked on a road network of Romania and the full OpenStreetMap graph of Atlanta.

---

## Why this project

Uninformed search is easy to write and expensive to run. The interesting engineering happens when you start pruning: a good heuristic, a search that meets in the middle, a termination condition that's provably safe to stop on. This project implements that progression end to end and measures the payoff at each step.

The tridirectional case is the fun one. It has no textbook pseudocode — you're searching from three sources simultaneously, each toward the other two, and you have to reason carefully about when it's safe to commit to a meeting point.

## Algorithms implemented

| Algorithm | Frontier | Heuristic | Notes |
|---|---|---|---|
| Breadth-first search | FIFO queue | — | Optimal only for unit edge weights; goal-tested on generation, not expansion |
| Uniform-cost search | Min-heap | — | Dijkstra with a single source/target |
| A\* | Min-heap | Euclidean | Admissible + consistent, so no reopening needed |
| Bidirectional UCS | 2 min-heaps | — | Alternates frontiers; stops when `top_f + top_b ≥ μ` |
| Bidirectional A\* | 2 min-heaps | Averaged | Uses a consistent potential function so both directions agree on edge costs |
| Tridirectional UCS | 3 min-heaps | — | Three sources, pairwise meeting points, cheapest two legs win |
| Tridirectional A\* | 3 min-heaps | Landmark-assisted | Each search is steered toward whichever of the other two goals looks closer |

### The tricky parts

**Bidirectional termination.** The naive instinct is to stop the moment the frontiers touch. That's wrong — the first meeting node is not necessarily on the shortest path. The correct condition is to keep expanding until the sum of the two frontier minimums exceeds the best path cost found so far, tracking `μ` (best-known cost) separately from the stopping test.

**Bidirectional A\* heuristic consistency.** Running a forward heuristic `h(n, goal)` and a backward heuristic `h(n, start)` independently breaks the termination proof, because the two searches are effectively optimizing different edge-cost functions. The fix is a balanced potential — `p(n) = (h_f(n) - h_b(n)) / 2` — which makes both directions consistent with respect to the same modified graph.

**Tridirectional is not three bidirectional searches.** Six frontiers is the obvious approach and also the wasteful one. Three frontiers suffice: each search expands outward once, and any node reached by two of them is a candidate meeting point for that pair. The final answer is the cheapest two of the three pairwise paths, joined at the shared node.

**Priority queue tie-breaking.** Equal priorities are broken FIFO via a monotonically increasing insertion counter pushed as the second tuple element. Without it, Python's heap falls through to comparing the payloads, which is both non-deterministic across runs and a `TypeError` waiting to happen on non-comparable values. The queue permits duplicate entries and uses lazy deletion rather than a decrease-key operation — amortized O(1) insert, O(log n) pop.

## Results

Node expansions on the Romania graph, averaged over all source/target pairs:

| Algorithm | Nodes expanded | Relative |
|---|---|---|
| Uniform-cost search | *TBD* | 1.00× |
| A\* (Euclidean) | *TBD* | — |
| Bidirectional UCS | *TBD* | — |
| Bidirectional A\* | *TBD* | — |

> Numbers are placeholders — drop in your measured counts from `search_unit_tests.py`.

Expansions are counted by an `ExplorableGraph` wrapper that instruments neighbor access, so the metric reflects real work done rather than a self-reported counter.

## Project structure

```
submission.py                    # All algorithm implementations
explorable_graph.py              # networkx wrapper that instruments node access
visualize_graph.py               # Renders paths and explored sets; GeoJSON for Atlanta
osm2networkx.py                  # OSM → networkx conversion
romania_graph.pickle             # Romania road network
atlanta_osm.pickle               # Atlanta OSM network
search_unit_tests.py             # All-pairs correctness + expansion-count checks
search_submission_tests.py       # Focused tests per algorithm
search_submission_tests_grid.py  # Uniform grid, visualizes frontier shape
```

## Running it

```bash
conda env create -f environment.yml
conda activate ai_env

python -m pytest search_unit_tests.py
```

Single test:

```python
import search_submission_tests as tests

t = tests.TestBasicSearch()
t.setUp()
t.test_bfs()
```

## Visualization

Romania renders directly in a matplotlib window with the path and explored set highlighted — useful for seeing the classic teardrop shape of A\*'s frontier versus the circle of UCS.

Atlanta is too large for that, so the bidirectional tests emit GeoJSON instead. Drop the output into [geojson.io](http://geojson.io/) to inspect it on a real map.

## References

- Korf, [Finding Optimal Solutions to Rubik's Cube Using Pattern Databases](https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf)
- Goldberg & Harrelson, [Computing the Shortest Path: A\* Search Meets Graph Theory](http://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/02-search-Goldberg03tr.pdf)
- Gutman, [Reach-based Routing for Road Networks](http://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/02-search-Gutman04siam.pdf)
- Goldberg et al., [Reach for A\*: An Efficient Point-to-Point Shortest Path Algorithm](http://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/02-search-01-Astart-ALT-Reach.pdf)
- [God's Number is 26 in the Quarter-Turn Metric](http://www.cube20.org/qtm/)

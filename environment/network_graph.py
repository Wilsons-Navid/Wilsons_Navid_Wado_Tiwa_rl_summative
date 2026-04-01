"""
Network topology generation for the Cyber Threat Hunting environment.
Generates realistic enterprise network graphs with different node types.
"""

import numpy as np
from enum import IntEnum


class NodeType(IntEnum):
    WORKSTATION = 0
    SERVER = 1
    DATABASE = 2
    FIREWALL = 3
    ROUTER = 4


class ThreatType(IntEnum):
    NONE = 0
    MALWARE = 1
    BACKDOOR = 2
    CRYPTOMINER = 3
    DATA_EXFILTRATION = 4


THREAT_PROPERTIES = {
    ThreatType.MALWARE: {
        "stealth": 0.3,        # easy to detect with quick scan
        "spread_rate": 0.15,   # can spread to neighbors
        "damage_rate": 2.0,    # moderate damage per step
    },
    ThreatType.BACKDOOR: {
        "stealth": 0.8,        # hard to detect, needs deep analysis
        "spread_rate": 0.0,    # doesn't spread
        "damage_rate": 1.0,    # slow but persistent
    },
    ThreatType.CRYPTOMINER: {
        "stealth": 0.5,        # moderate detection difficulty
        "spread_rate": 0.1,    # slight spread
        "damage_rate": 1.5,    # resource drain
    },
    ThreatType.DATA_EXFILTRATION: {
        "stealth": 0.7,        # fairly stealthy
        "spread_rate": 0.0,    # doesn't spread
        "damage_rate": 3.0,    # high damage (data loss)
    },
}


def generate_network(num_nodes=14, num_threats=4, seed=None):
    """Generate a realistic enterprise network topology.

    Returns:
        adjacency: (num_nodes, num_nodes) binary adjacency matrix
        node_types: (num_nodes,) array of NodeType values
        threats: dict mapping node_index -> ThreatType (for infected nodes)
    """
    rng = np.random.default_rng(seed)

    # Assign node types with realistic distribution
    node_types = np.zeros(num_nodes, dtype=np.int32)
    # First node is always a firewall (entry point)
    node_types[0] = NodeType.FIREWALL
    # One or two routers
    router_indices = rng.choice(range(1, num_nodes), size=2, replace=False)
    node_types[router_indices] = NodeType.ROUTER
    # Two databases
    remaining = [i for i in range(1, num_nodes) if i not in router_indices]
    db_indices = rng.choice(remaining, size=2, replace=False)
    node_types[db_indices] = NodeType.DATABASE
    # Three servers
    remaining = [i for i in remaining if i not in db_indices]
    server_indices = rng.choice(remaining, size=min(3, len(remaining)), replace=False)
    node_types[server_indices] = NodeType.SERVER
    # Rest are workstations (default 0)

    # Build adjacency matrix — tree-like with extra edges for realism
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    # Connect firewall to routers
    for r in router_indices:
        adjacency[0, r] = 1
        adjacency[r, 0] = 1

    # Connect each remaining node to at least one existing connected node
    connected = {0, *router_indices.tolist()}
    unconnected = [i for i in range(1, num_nodes) if i not in connected]
    rng.shuffle(unconnected)

    for node in unconnected:
        # Connect to a random already-connected node
        parent = rng.choice(list(connected))
        adjacency[node, parent] = 1
        adjacency[parent, node] = 1
        connected.add(node)

    # Add extra edges for realism (not a pure tree)
    num_extra = rng.integers(3, 6)
    for _ in range(num_extra):
        a, b = rng.choice(num_nodes, size=2, replace=False)
        if adjacency[a, b] == 0:
            adjacency[a, b] = 1
            adjacency[b, a] = 1

    # Place threats on non-firewall nodes
    threat_candidates = [i for i in range(num_nodes) if node_types[i] != NodeType.FIREWALL]
    num_threats = min(num_threats, len(threat_candidates))
    infected_nodes = rng.choice(threat_candidates, size=num_threats, replace=False)

    threat_types_available = [ThreatType.MALWARE, ThreatType.BACKDOOR,
                              ThreatType.CRYPTOMINER, ThreatType.DATA_EXFILTRATION]
    threats = {}
    for node in infected_nodes:
        threats[int(node)] = rng.choice(threat_types_available)

    return adjacency, node_types, threats


def get_neighbors(adjacency, node):
    """Get list of neighbor indices for a given node."""
    return list(np.where(adjacency[node] == 1)[0])

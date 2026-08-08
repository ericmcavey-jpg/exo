import os
from collections.abc import Generator, Mapping

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo
from exo.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)

# Placement previously validated against each node's raw ram_available, but
# the worker separately caps actual Metal usage via mx.set_wired_limit() to
# a fraction of live available RAM (see set_wired_limit_for_model() in
# utils_mlx.py) -- these two numbers must agree, or placement can approve an
# allocation the runtime will then refuse (this happened in production,
# 2026-07-08 DeepSeek-V3.2-4bit bakeoff: a node with 88GB "available" was
# approved for 75GB of layers, but the worker's own separately-configured
# ceiling only allowed ~78GB total, leaving far less real margin than
# placement assumed). Keep this fraction identical to
# EXO_WIRED_LIMIT_FRACTION's default so both layers agree by default.
_WIRED_LIMIT_FRACTION = float(os.environ.get("EXO_WIRED_LIMIT_FRACTION", "0.97"))

# mx.set_wired_limit() *hard-enforces* mx.device_info()'s
# max_recommended_working_set_size as an upper bound (confirmed 2026-07-08 --
# requesting more raises "Setting a wired limit larger than the maximum
# working set size is not allowed" and crashes the runner). Apple's own
# per-chip default for that value proved too conservative in practice
# (DeepSeek-V3.2-4bit bakeoff, 2026-07-08), so each node's
# iogpu.wired_limit_mb sysctl is now deliberately overridden higher (see
# fleet runbook) -- this raises what mx.device_info() reports in turn. The
# master has no direct mx.device_info() access (it aggregates
# worker-reported MemoryUsage, not GPU device metadata), so this is a
# hardcoded lookup for the two chips in this fleet, keyed by total RAM.
# THESE VALUES MUST BE KEPT IN SYNC with whatever iogpu.wired_limit_mb is
# actually set to on each node -- if you change one, change the other.
# Current values (97% of live available RAM at time of setting, conservative
# node in each tier, 2026-07-08):
#   M3 Ultra, 96GB total   -> 82.47 GiB
#   M3/M4 Max, 128GB total -> 115.59 GiB
_DEVICE_RECOMMENDED_MAX_GIB_BY_TOTAL_GB = [
    # 2026-07-12: lowered from 115.59/82.47 to leave a fixed ~17 GiB Metal
    # shader-spike buffer below physical. GLM-5.2 4-node load OOM'd on nodes
    # with only ~11 GiB buffer (m3b 724x) while >=17 GiB nodes were clean.
    # MUST stay in sync with exo_launch_bigmodel.sh RESERVE_GIB (default 17).
    # 2026-08-03: added a 256GB (M3 Ultra, e.g. m3c) tier. Before this, a 256GB
    # node fell into the >=100 bucket and was capped at 111 GiB -- treated as a
    # 128GB node -- so proportional allocation gave it ~22% of layers instead of
    # its rightful ~36%, wasting ~120 GiB and choking the 96GB nodes at prefill
    # peak. 232.0 matches m3c's already-set iogpu.wired_limit_mb (237733 = 232 GiB),
    # leaving a ~24 GiB buffer below its 256 GiB physical.
    # 2026-08-07: 232.0 -> 239.0. The 232.0 was pinned to an iogpu.wired_limit_mb of
    # 237733 that no longer exists: exo_launch_bigmodel.sh's retune step now sets 249344
    # (243.5 GiB) on the 256GB nodes every launch, so this table -- which this comment
    # block itself says MUST stay in sync with the sysctl -- had drifted ~11 GiB BELOW
    # reality. Because _effective_available() takes min(0.97 * live_available, this cap),
    # the stale constant silently became the binding constraint: DeepSeek-V4-Pro (791 GiB,
    # 61 layers) PINFAILed with "m3c requires 233.51 GB for 18 layers, but only has 232.00
    # GB available" while m3c actually had 245 GiB free and a 243.5 GiB ceiling. Nothing
    # observable moved it -- purge, killing the wallpaper stack, raising the sysctl, and a
    # full fleet restart all left it at exactly 232.00, because it was never reading any of
    # them. 239.0 = 256 - 17, keeping the 17 GiB Metal shader-spike buffer the note above
    # earned the hard way, and staying below the live sysctl so placement remains the
    # conservative side of the pair.
    (200, 239.0),  # >=200GB total (256GB nodes): matches iogpu.wired_limit_mb
    (100, 111.0),  # >=100GB total (128GB nodes): 128 - 17
    (0, 79.0),     # <100GB total (96GB nodes): 96 - 17
]


def _device_recommended_max(ram_total: Memory) -> Memory:
    for threshold_gb, max_gib in _DEVICE_RECOMMENDED_MAX_GIB_BY_TOTAL_GB:
        if ram_total.in_gb >= threshold_gb:
            return Memory.from_bytes(int(max_gib * 1024**3))
    return Memory.from_bytes(int(_DEVICE_RECOMMENDED_MAX_GIB_BY_TOTAL_GB[-1][1] * 1024**3))


def _effective_available(mem_usage: MemoryUsage) -> Memory:
    """This node's ram_available, scaled down to match what the worker will
    actually enforce at runtime via mx.set_wired_limit() -- the smaller of
    (a fraction of live available RAM) and (this chip's hard MLX ceiling)."""
    fraction_based = int(mem_usage.ram_available.in_bytes * _WIRED_LIMIT_FRACTION)
    device_cap = _device_recommended_max(mem_usage.ram_total).in_bytes
    return Memory.from_bytes(min(fraction_based, device_cap))


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
) -> list[Cycle]:
    filtered_cycles: list[Cycle] = []
    for cycle in cycles:
        if not all(node in node_memory for node in cycle):
            continue

        total_mem = sum(
            (
                _effective_available(node_memory[node_id])
                for node_id in cycle.node_ids
            ),
            start=Memory(),
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
) -> list[int]:
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )

    # Largest remainder: floor each, then distribute remainder by fractional part
    raw = [f * total_layers for f in memory_fractions]
    result = [int(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
    for i in range(total_layers - sum(result)):
        result[by_remainder[i]] += 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Memory:
    total_memory = sum(
        (
            _effective_available(node_memory[node_id])
            for node_id in node_ids
        ),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")
    return total_memory


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
) -> list[int]:
    layer_allocations = allocate_layers_proportionally(
        total_layers=model_card.n_layers,
        memory_fractions=[
            _effective_available(node_memory[node_id]) / total_memory
            for node_id in node_ids
        ],
    )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        available_memory = _effective_available(node_memory[node_id])
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but only has {available_memory.in_gb:.2f} GB available "
                f"(effective, {_WIRED_LIMIT_FRACTION:.0%} of live available)"
            )

    return layer_allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(model_card, cycle, node_memory)
    else:
        return _get_shard_assignments_for_pure_pipeline(model_card, cycle, node_memory)


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(pipeline_node_ids, node_memory)
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids, node_memory, pipeline_memory, model_card
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory)

    layer_allocations = _allocate_and_validate_layers(
        cycle.node_ids, node_memory, total_memory, model_card
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j."""
    import ipaddress as _ipa
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            _ip = connection.sink_multiaddr.ip_address
            # PATCH 2026-08-05: never offer a loopback / link-local / unspecified address
            # as a ring connection IP. fe80::1%lo0 (loopback link-local) was discovered as a
            # SocketConnection sink and won selection for a NEIGHBOUR pair (its TB peer conn
            # was absent from the candidate set), so activations routed to loopback and decode
            # hung with zero tokens. Filtering here means a genuinely unroutable neighbour pair
            # raises loudly at placement ("requires connectivity between neighbouring nodes")
            # instead of silently hanging.
            _bare = _ip.split("%", 1)[0]
            try:
                _a = _ipa.ip_address(_bare)
                if _a.is_loopback or _a.is_link_local or _a.is_unspecified:
                    continue
            except ValueError:
                pass
            yield _ip


def find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Priority: ethernet > wifi > unknown > thunderbolt
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }

    # Ring should prioritise fastest connection. As a best-effort, we prioritise TB.
    # TODO: Profile and get actual connection speeds.
    if ring:
        priority = {
            "thunderbolt": 0,
            "maybe_ethernet": 1,
            "ethernet": 2,
            "wifi": 3,
            "unknown": 4,
        }

    # JACCL/RDMA coordinator traffic should stay on the fastest direct fabric.
    else:
        priority = {
            "thunderbolt": 0,
            "maybe_ethernet": 1,
            "ethernet": 2,
            "wifi": 3,
            "unknown": 4,
        }
    return min(ips, key=lambda ip: priority.get(ip_to_type.get(ip, "unknown"), 2))


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }

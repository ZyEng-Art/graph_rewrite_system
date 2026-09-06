from __future__ import annotations

from model import S0ActionBindingModel
from paged_model import PagedActionBindingModel


def build_model(rules, num_xfers: int, args: dict):
    architecture = args.get("architecture", "legacy")
    if architecture == "paged_action":
        return PagedActionBindingModel(
            rules,
            num_xfers=num_xfers,
            width=args["width"],
            retrieval_width=args["retrieval_width"],
            graph_layers=args["graph_layers"],
            action_layers=args.get("action_layers", 4),
            action_heads=args.get("action_heads", 6),
            max_sequence_length=args.get("max_sequence_length", 256),
            ordered_binding_roles=args.get("ordered_binding_roles", False),
            readout_graph_layers=args.get("readout_graph_layers", 0),
            readout_graph_input=args.get("readout_graph_input", "cached"),
            readout_locality_features=args.get(
                "readout_locality_features", False
            ),
            identity_readout_prefix=args.get("identity_readout_prefix", 0),
            readout_attention_backend=args.get(
                "readout_attention_backend", "sdpa"
            ),
            action_value_head=args.get("action_value_head", False),
            source_topology_layers=args.get("source_topology_layers", 0),
            source_id_frequency_prior=args.get("source_id_frequency_prior", 0.0),
        )
    if architecture != "legacy":
        raise ValueError(f"unknown architecture: {architecture}")
    return S0ActionBindingModel(
        rules,
        num_xfers=num_xfers,
        width=args["width"],
        retrieval_width=args["retrieval_width"],
        graph_layers=args["graph_layers"],
        current_graph_layers=args.get("current_graph_layers", 5),
        use_action_history=not args.get("state_only", False),
        use_locality_features=args.get("locality_features", False),
        source_topology_layers=args.get("source_topology_layers", 0),
        source_id_frequency_prior=args.get("source_id_frequency_prior", 0.0),
    )

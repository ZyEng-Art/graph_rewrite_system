from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ACTOR_SHA256 = "bc0bc79503b49ed2a53f945d944dbe9c9d86bb45cc42a8a8a95e4b32d2e3d11e"
PPO_SHA256 = "a387e375ec50383b67d730b520b191b2818ed64f622fc6a8d9e6445a9604e38d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def replace_last(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count < 1:
        raise RuntimeError(f"expected at least one {label} anchor, found {count}")
    position = source.rfind(old)
    return source[:position] + new + source[position + len(old) :]


def instrument_actor(path: Path) -> None:
    actual = sha256(path)
    if actual != ACTOR_SHA256:
        raise RuntimeError(f"unexpected actor.py SHA-256: {actual}")
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "from model.actor_critic import ActorCritic\n",
        "from model.actor_critic import ActorCritic\n"
        "from original_quarl_rollout_profiler import RolloutProfiler\n",
        "profiler import",
    )
    source = replace_once(
        source,
        "        self.init_buffer_turn: int = 0\n",
        "        self.init_buffer_turn: int = 0\n"
        "        self.rollout_profiler = RolloutProfiler(self.device, self.id)\n",
        "profiler initialization",
    )

    anchors = [
        (
            "        dgl_graphs: List[dgl.DGLGraph] = [g.to_dgl_graph() for g in cur_graphs]\n",
            "        profile_started = self.rollout_profiler.start()\n"
            "        dgl_graphs: List[dgl.DGLGraph] = [g.to_dgl_graph() for g in cur_graphs]\n"
            "        self.rollout_profiler.stop('inference.graph_to_dgl', profile_started)\n",
            "graph to DGL",
        ),
        (
            "        b_state: dgl.DGLGraph = dgl.batch(dgl_graphs).to(self.device)\n"
            "        num_nodes: torch.LongTensor = (\n"
            "            b_state.batch_num_nodes()\n"
            "        )  # (num_graphs, ) assert each elem > 0\n",
            "        profile_started = self.rollout_profiler.start()\n"
            "        b_state: dgl.DGLGraph = dgl.batch(dgl_graphs).to(self.device)\n"
            "        num_nodes: torch.LongTensor = (\n"
            "            b_state.batch_num_nodes()\n"
            "        )  # (num_graphs, ) assert each elem > 0\n"
            "        self.rollout_profiler.stop(\n"
            "            'inference.dgl_batch_h2d', profile_started, cuda=True\n"
            "        )\n",
            "DGL batch",
        ),
        (
            "        b_node_embeds: torch.Tensor = self.ac_net.gnn(b_state)\n",
            "        profile_started = self.rollout_profiler.start()\n"
            "        b_node_embeds: torch.Tensor = self.ac_net.gnn(b_state)\n"
            "        self.rollout_profiler.stop('inference.gnn', profile_started, cuda=True)\n",
            "GNN",
        ),
        (
            "        b_node_values: torch.Tensor = self.ac_net.critic(b_node_embeds).squeeze()\n",
            "        profile_started = self.rollout_profiler.start()\n"
            "        b_node_values: torch.Tensor = self.ac_net.critic(b_node_embeds).squeeze()\n"
            "        self.rollout_profiler.stop('inference.critic', profile_started, cuda=True)\n"
            "        profile_started = self.rollout_profiler.start()\n",
            "critic",
        ),
        (
            "        sampled_node_embeds = b_node_embeds[sampled_node_b_ids]\n"
            "        \"\"\"use Actor to evaluate xfers for sampled nodes\"\"\"\n",
            "        sampled_node_embeds = b_node_embeds[sampled_node_b_ids]\n"
            "        self.rollout_profiler.stop(\n"
            "            'inference.node_sampling', profile_started, cuda=True\n"
            "        )\n"
            "        \"\"\"use Actor to evaluate xfers for sampled nodes\"\"\"\n"
            "        profile_started = self.rollout_profiler.start()\n",
            "node sampling",
        ),
        (
            "        xfer_logits: torch.Tensor = self.ac_net.actor(sampled_node_embeds)\n"
            "        \"\"\"sample action_xfer with mask\"\"\"\n",
            "        xfer_logits: torch.Tensor = self.ac_net.actor(sampled_node_embeds)\n"
            "        self.rollout_profiler.stop('inference.actor', profile_started, cuda=True)\n"
            "        \"\"\"sample action_xfer with mask\"\"\"\n"
            "        profile_started = self.rollout_profiler.start()\n",
            "actor",
        ),
        (
            "        # end for\n"
            "        softmax_xfer_logits = masked_softmax(xfer_logits, av_xfer_masks)\n",
            "        # end for\n"
            "        self.rollout_profiler.stop(\n"
            "            'inference.available_xfers', profile_started, cuda=True\n"
            "        )\n"
            "        profile_started = self.rollout_profiler.start()\n"
            "        softmax_xfer_logits = masked_softmax(xfer_logits, av_xfer_masks)\n",
            "available xfers",
        ),
        (
            "        action_node_values: List[float] = b_node_values_pad[\n"
            "            list(range(num_eps)), action_nodes\n"
            "        ].tolist()\n"
            "        return (\n",
            "        action_node_values: List[float] = b_node_values_pad[\n"
            "            list(range(num_eps)), action_nodes\n"
            "        ].tolist()\n"
            "        self.rollout_profiler.stop(\n"
            "            'inference.xfer_sampling_transfer', profile_started, cuda=True\n"
            "        )\n"
            "        return (\n",
            "xfer sampling",
        ),
        (
            "        buffer_idx_list: List[int] = []\n",
            "        profile_started = self.rollout_profiler.start()\n"
            "        buffer_idx_list: List[int] = []\n",
            "root sampling start",
        ),
        (
            "        # end for\n"
            "        \"\"\"communicate with other ranks to get max of max_eps_len_for_all\"\"\"\n"
            "        max_eps_len_all_ranks = torch.zeros(self.num_agents).to(self.device)\n",
            "        # end for\n"
            "        self.rollout_profiler.stop('setup.sample_roots', profile_started)\n"
            "        \"\"\"communicate with other ranks to get max of max_eps_len_for_all\"\"\"\n"
            "        profile_started = self.rollout_profiler.start()\n"
            "        max_eps_len_all_ranks = torch.zeros(self.num_agents).to(self.device)\n",
            "root sampling end",
        ),
        (
            "        max_eps_len_for_all = int(max_eps_len_all_ranks.max())\n\n"
            "        \"\"\"run episodes\"\"\"\n",
            "        max_eps_len_for_all = int(max_eps_len_all_ranks.max())\n"
            "        self.rollout_profiler.stop(\n"
            "            'setup.horizon_sync', profile_started, cuda=True\n"
            "        )\n\n"
            "        \"\"\"run episodes\"\"\"\n",
            "horizon sync",
        ),
        (
            "                dgl_graphs += mb_dgl_graphs\n",
            "                profile_started = self.rollout_profiler.start()\n"
            "                dgl_graphs += mb_dgl_graphs\n",
            "inference merge start",
        ),
        (
            "                av_xfer_masks = cast(\n"
            "                    torch.BoolTensor, torch.cat([av_xfer_masks, mb_av_xfer_masks])\n"
            "                )\n",
            "                av_xfer_masks = cast(\n"
            "                    torch.BoolTensor, torch.cat([av_xfer_masks, mb_av_xfer_masks])\n"
            "                )\n"
            "                self.rollout_profiler.stop(\n"
            "                    'inference.result_merge', profile_started\n"
            "                )\n",
            "inference merge end",
        ),
        (
            "                next_graph, next_nodes = graph.apply_xfer_with_local_state_tracking(\n",
            "                profile_started = self.rollout_profiler.start()\n"
            "                next_graph, next_nodes = graph.apply_xfer_with_local_state_tracking(\n",
            "apply start",
        ),
        (
            "                    predecessor_layers=self.xfer_pred_layers,\n"
            "                )\n"
            "                \"\"\"parse result, compute reward\"\"\"\n",
            "                    predecessor_layers=self.xfer_pred_layers,\n"
            "                )\n"
            "                self.rollout_profiler.stop(\n"
            "                    'environment.apply_xfer', profile_started\n"
            "                )\n"
            "                \"\"\"parse result, compute reward\"\"\"\n"
            "                profile_started = self.rollout_profiler.start()\n",
            "apply end",
        ),
        (
            "                if i_step - last_eps_end >= max_eps_len_for_all:\n"
            "                    game_over = True  # exceed len limit\n\n"
            "                \"\"\"collect data\"\"\"\n",
            "                if i_step - last_eps_end >= max_eps_len_for_all:\n"
            "                    game_over = True  # exceed len limit\n"
            "                self.rollout_profiler.stop(\n"
            "                    'environment.reward_termination', profile_started\n"
            "                )\n\n"
            "                \"\"\"collect data\"\"\"\n"
            "                profile_started = self.rollout_profiler.start()\n",
            "reward",
        ),
        (
            "                eps_list.next_nodes.append(next_nodes)\n\n"
            "                # collect cur_graph info\n",
            "                eps_list.next_nodes.append(next_nodes)\n"
            "                self.rollout_profiler.stop(\n"
            "                    'experience.next_state', profile_started\n"
            "                )\n\n"
            "                # collect cur_graph info\n"
            "                profile_started = self.rollout_profiler.start()\n",
            "next state",
        ),
        (
            "                eps_list.action.append(action)\n\n"
            "                # collect other info\n",
            "                eps_list.action.append(action)\n"
            "                self.rollout_profiler.stop(\n"
            "                    'experience.current_state', profile_started\n"
            "                )\n\n"
            "                # collect other info\n"
            "                profile_started = self.rollout_profiler.start()\n",
            "current state",
        ),
        (
            "                eps_list.info.append({})\n\n"
            "                \"\"\"collect info for graph buffer\"\"\"\n",
            "                eps_list.info.append({})\n"
            "                self.rollout_profiler.stop(\n"
            "                    'experience.append', profile_started\n"
            "                )\n\n"
            "                \"\"\"collect info for graph buffer\"\"\"\n"
            "                profile_started = self.rollout_profiler.start()\n",
            "experience append",
        ),
        (
            "                    # end if better\n"
            "                # end if\n\n"
            "                if i_step == max_eps_len_for_all - 1:\n",
            "                    # end if better\n"
            "                # end if\n"
            "                self.rollout_profiler.stop(\n"
            "                    'buffer.update_and_best', profile_started\n"
            "                )\n\n"
            "                profile_started = self.rollout_profiler.start()\n"
            "                if i_step == max_eps_len_for_all - 1:\n",
            "buffer update",
        ),
        (
            "                else:\n"
            "                    cur_graphs[i_eps] = next_graph\n"
            "            # end for i_eps\n",
            "                else:\n"
            "                    cur_graphs[i_eps] = next_graph\n"
            "                self.rollout_profiler.stop(\n"
            "                    'environment.restart_or_advance', profile_started\n"
            "                )\n"
            "            # end for i_eps\n",
            "restart",
        ),
        (
            "        eps_list_cat = ExperienceList.new_empty()\n",
            "        profile_started = self.rollout_profiler.start()\n"
            "        eps_list_cat = ExperienceList.new_empty()\n",
            "finalize start",
        ),
        (
            "        eps_list_cat.sanity_check()\n"
            "        return eps_list_cat\n",
            "        eps_list_cat.sanity_check()\n"
            "        self.rollout_profiler.stop('finalize.concatenate', profile_started)\n"
            "        return eps_list_cat\n",
            "finalize end",
        ),
    ]
    for old, new, label in anchors:
        if label in {"GNN", "critic"}:
            source = replace_last(source, old, new, label)
        else:
            source = replace_once(source, old, new, label)
    path.write_text(source, encoding="utf-8")


def instrument_ppo(path: Path) -> None:
    actual = sha256(path)
    if actual != PPO_SHA256:
        raise RuntimeError(f"unexpected ppo.py SHA-256: {actual}")
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "        s_time_collect = get_time_ns()\n"
        "        self.agent.perpare_buf_for_next_iter()\n",
        "        s_time_collect = get_time_ns()\n"
        "        self.agent.rollout_profiler.begin(self.i_iter)\n"
        "        profile_started = self.agent.rollout_profiler.start()\n"
        "        self.agent.perpare_buf_for_next_iter()\n"
        "        self.agent.rollout_profiler.stop('buffer.prepare', profile_started)\n",
        "rollout begin",
    )
    source = replace_once(
        source,
        "        dur_s_collect = dur_ms(e_time_collect, s_time_collect) / 1e3\n"
        "        self.tot_exps_collected += len(exp_list)\n",
        "        dur_s_collect = dur_ms(e_time_collect, s_time_collect) / 1e3\n"
        "        self.agent.rollout_profiler.finish(\n"
        "            dur_s_collect, len(exp_list), self.agent.graph_buffers\n"
        "        )\n"
        "        self.tot_exps_collected += len(exp_list)\n",
        "rollout finish",
    )
    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instrument the audited original Quarl rollout snapshot"
    )
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--profiler-module",
        type=Path,
        default=Path(__file__).parent
        / "quartz_patches"
        / "original_quarl_rollout_profiler.py",
    )
    args = parser.parse_args()
    actor_path = args.snapshot / "actor.py"
    ppo_path = args.snapshot / "ppo.py"
    instrument_actor(actor_path)
    instrument_ppo(ppo_path)
    destination = args.snapshot / "original_quarl_rollout_profiler.py"
    destination.write_bytes(args.profiler_module.read_bytes())
    print(f"instrumented {args.snapshot}")
    print(f"actor.py {sha256(actor_path)}")
    print(f"ppo.py {sha256(ppo_path)}")


if __name__ == "__main__":
    main()

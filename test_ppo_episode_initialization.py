from __future__ import annotations

from train_paged_ppo import group_episode_starts


def main() -> None:
    starts = ["root-a", None, "root-a", "root-b", None]
    unique, indices = group_episode_starts(starts, "deduplicated")
    assert unique == ["root-a", None, "root-b"]
    assert indices == [0, 1, 0, 2, 1]
    assert [unique[index] for index in indices] == starts

    duplicated, duplicated_indices = group_episode_starts(starts, "duplicated")
    assert duplicated == starts
    assert duplicated_indices == list(range(len(starts)))

    try:
        group_episode_starts(starts, "unknown")
    except ValueError as error:
        assert "unknown episode initialization backend" in str(error)
    else:
        raise AssertionError("unknown initialization backend was accepted")

    print("PPO episode start grouping preserves the original episode order")


if __name__ == "__main__":
    main()

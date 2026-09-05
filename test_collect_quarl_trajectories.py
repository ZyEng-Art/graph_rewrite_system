from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from collect_quarl_trajectories import (
    discover_trajectory_directories,
    excluded,
    parse_trajectory_directory,
)


def write_path(path: Path, names: list[str]) -> None:
    path.mkdir(parents=True)
    for name in names:
        (path / name).write_text("OPENQASM 2.0;\n")


class QuarlTrajectoryParsingTest(unittest.TestCase):
    def test_parse_and_discover_saved_trajectory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "barenco_tof_3" / "38_3"
            write_path(
                selected,
                [
                    "1_41_0_7_896.qasm",
                    "0_39_-2_5_40.qasm",
                    "2_41_0_0_0.qasm",
                ],
            )
            write_path(root / "single", ["0_39_0_0_0.qasm"])

            self.assertEqual(discover_trajectory_directories([root]), [selected])
            rows = parse_trajectory_directory(selected)
            self.assertEqual([row.step for row in rows], [0, 1, 2])
            self.assertEqual(rows[0].xfer_id, 40)
            self.assertEqual(rows[-1].xfer_id, 0)
            self.assertTrue(excluded(selected, ["barenco_tof_3/38_3"]))

    def test_rejects_noncontiguous_steps(self) -> None:
        with TemporaryDirectory() as directory:
            selected = Path(directory) / "broken"
            write_path(
                selected,
                ["0_39_0_1_40.qasm", "2_38_0_0_0.qasm"],
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                parse_trajectory_directory(selected)


if __name__ == "__main__":
    unittest.main()

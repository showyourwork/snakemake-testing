from pathlib import Path
from typing import List

import pytest
from snakemake_testing.runner import TemporaryDirectory, run_snakemake


@pytest.mark.parametrize("force_explicit", [True, False])
def test_temporary_directory_cleanup(force_explicit: bool) -> None:
    tempdir = TemporaryDirectory("", force_explicit=force_explicit)
    assert tempdir.name.is_dir()
    tempdir.cleanup()
    assert not tempdir.name.is_dir()
    assert not tempdir.name.exists()


def test_simple():
    run_snakemake("tests/projects/simple")


def test_simple_cleanup():
    # Here we want to check that the temporary directory actually gets properly
    # cleaned up when the test results go out of scope so we put it in a
    # function and call that, keeping track of the temporary directories that
    # were created
    reg: List[Path] = []

    def impl():
        tempdir = run_snakemake("tests/projects/simple")
        reg.append(tempdir._directory.name)

    impl()
    assert len(reg) == 1
    for tempdir in reg:
        assert not tempdir.is_dir()
        assert not tempdir.exists()


def test_simple_context():
    with run_snakemake("tests/projects/simple") as tempdir:
        assert tempdir.is_dir()
        assert (tempdir / "output1.txt").is_file()
        assert (tempdir / "output2.txt").is_file()
    assert not tempdir.is_dir()
    assert not tempdir.exists()

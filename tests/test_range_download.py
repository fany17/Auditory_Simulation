from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from scripts.public_s3_range_download import (
    MAX_WORKERS,
    ObjectPlan,
    RangeChunk,
    assemble_object,
    chunk_is_reusable,
    chunk_path,
    chunk_ranges,
    download_chunk,
    interleaved_tasks,
    load_plans,
    move_to_backup,
    object_chunks_ready,
    safe_mib_per_second,
    safe_destination,
    validate_inventory_rows,
    validate_range_response,
)
from scripts.public_s3_range_smoke import parse_content_range
from scripts.move_interrupted_partials import move_interrupted_partials


ROOT = Path(__file__).resolve().parents[1]


class FakeRangeResponse:
    def __init__(self, body: bytes, content_range: str) -> None:
        self.status = 206
        self.headers = {"Content-Range": content_range}
        self._body = BytesIO(body)

    def __enter__(self) -> FakeRangeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


def controlled_sandbox() -> tuple[Path, Path, Path]:
    root = Path(mkdtemp(prefix="m6a-range-controlled-"))
    dataset = root / "dataset"
    staging = root / "staging"
    dataset.mkdir()
    staging.mkdir()
    return root, dataset, staging


def tiny_plan() -> ObjectPlan:
    return ObjectPlan(
        relative_path="sub-SD011/file.edf",
        source_url="https://s3.amazonaws.com/openneuro.org/ds004703/sub-SD011/file.edf",
        expected_bytes=5,
        modified_at_utc="2026-08-11T00:00:00Z",
        chunks=(RangeChunk(0, 2, 5), RangeChunk(3, 4, 5)),
    )


class RangeDownloadTests(unittest.TestCase):
    def test_smoke_content_range_parser_is_strict(self) -> None:
        self.assertEqual(parse_content_range("bytes 0-1048575/1833841408"), (0, 1048575, 1833841408))
        for value in (None, "bytes 0-10/*", "0-10/100", "bytes 10-0/100"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_content_range(value)

    def test_chunk_ranges_are_fixed_nonoverlapping_and_cover_total(self) -> None:
        chunks = chunk_ranges(total_bytes=35, chunk_bytes=16)
        self.assertEqual(
            [(item.start, item.end, item.expected_bytes) for item in chunks],
            [(0, 15, 16), (16, 31, 16), (32, 34, 3)],
        )
        self.assertEqual(sum(item.expected_bytes for item in chunks), 35)

    def test_response_requires_206_exact_range_and_total(self) -> None:
        chunk = RangeChunk(16, 31, 35)
        validate_range_response(206, "bytes 16-31/35", chunk)
        for status, value in (
            (200, "bytes 16-31/35"),
            (206, "bytes 16-30/35"),
            (206, "bytes 16-31/36"),
            (206, None),
        ):
            with self.subTest(status=status, value=value):
                with self.assertRaises(OSError):
                    validate_range_response(status, value, chunk)

    def test_safe_destination_rejects_escape(self) -> None:
        root = ROOT / "tests" / "fixtures"
        self.assertEqual(safe_destination(root, "range_chunk_5_bytes.txt").parent, root.resolve())
        with self.assertRaises(ValueError):
            safe_destination(root, "../escape")

    def test_existing_chunk_is_reused_only_by_exact_expected_size(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "range_chunk_5_bytes.txt"
        self.assertTrue(chunk_is_reusable(fixture, RangeChunk(0, 4, 5)))
        self.assertFalse(chunk_is_reusable(fixture, RangeChunk(0, 5, 6)))

    def test_inventory_rejects_duplicates_empty_unsafe_and_nonofficial_sources(self) -> None:
        valid = {
            "path": "sub-SD011/file.edf",
            "bytes": "5",
            "modified_at_utc": "2026-08-11T00:00:00Z",
            "source_url": "https://s3.amazonaws.com/openneuro.org/ds004703/sub-SD011/file.edf",
        }
        self.assertEqual(list(validate_inventory_rows([valid])), ["sub-SD011/file.edf"])
        invalid_rows = (
            [valid, dict(valid)],
            [{**valid, "path": ""}],
            [{**valid, "path": "../escape"}],
            [{**valid, "source_url": ""}],
            [{**valid, "source_url": "https://example.invalid/ds004703/sub-SD011/file.edf"}],
            [{**valid, "source_url": "http://s3.amazonaws.com/openneuro.org/ds004703/sub-SD011/file.edf"}],
            [{**valid, "source_url": "https://s3.amazonaws.com/openneuro.org/ds004703/sub-SD019/file.edf"}],
            [{**valid, "bytes": "0"}],
            [{**valid, "bytes": "-1"}],
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    validate_inventory_rows(rows)

    def test_repository_inventory_accepts_exact_official_sd011_entry(self) -> None:
        relative = "sub-SD011/ses-01/ieeg/sub-SD011_ses-01_task-PassiveListen_ieeg.edf"
        plans = load_plans(
            ROOT / "reports" / "ds004703_s3_inventory.csv",
            [relative],
            16 * 1024 * 1024,
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].relative_path, relative)
        self.assertEqual(plans[0].expected_bytes, 1833841408)

    def test_object_readiness_fails_if_any_chunk_is_missing(self) -> None:
        staging = ROOT / "tests" / "fixtures"
        plan = ObjectPlan(
            relative_path="range_chunk_5_bytes.txt",
            source_url="https://example.invalid/object",
            expected_bytes=5,
            modified_at_utc="2026-08-11T00:00:00Z",
            chunks=(RangeChunk(0, 4, 5),),
        )
        self.assertFalse(object_chunks_ready(plan, staging))

    def test_interleaving_spreads_workers_across_objects(self) -> None:
        plans = [
            ObjectPlan(
                relative_path=f"object-{index}",
                source_url="https://example.invalid/object",
                expected_bytes=4,
                modified_at_utc="2026-08-11T00:00:00Z",
                chunks=chunk_ranges(4, 2),
            )
            for index in range(3)
        ]
        tasks = interleaved_tasks(plans)
        self.assertEqual([item[0].relative_path for item in tasks[:3]], ["object-0", "object-1", "object-2"])
        self.assertEqual(MAX_WORKERS, 8)

    def test_controlled_temp_multichunk_assembly_is_ordered_and_atomic(self) -> None:
        _, dataset, staging = controlled_sandbox()
        plan = tiny_plan()
        payloads = (b"ABC", b"DE")
        for chunk, payload in zip(plan.chunks, payloads):
            path = chunk_path(staging, plan.relative_path, chunk)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        result = assemble_object(plan, dataset, staging, "test-assembly")
        self.assertEqual(result["status"], "ASSEMBLED")
        final = dataset / plan.relative_path
        self.assertEqual(final.read_bytes(), b"ABCDE")
        self.assertEqual(list(final.parent.glob(final.name + ".partial-range-*")), [])

    def test_controlled_temp_missing_chunk_blocks_assembly(self) -> None:
        _, dataset, staging = controlled_sandbox()
        plan = tiny_plan()
        first = chunk_path(staging, plan.relative_path, plan.chunks[0])
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"ABC")
        result = assemble_object(plan, dataset, staging, "test-missing")
        self.assertEqual(result["status"], "ASSEMBLY_BLOCKED_INCOMPLETE_CHUNKS")
        self.assertFalse((dataset / plan.relative_path).exists())

    def test_controlled_temp_mismatched_final_and_partial_are_backed_up(self) -> None:
        _, dataset, staging = controlled_sandbox()
        plan = tiny_plan()
        for chunk, payload in zip(plan.chunks, (b"ABC", b"DE")):
            path = chunk_path(staging, plan.relative_path, chunk)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        final = dataset / plan.relative_path
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"X")
        stale = final.with_name(final.name + ".partial-range-old")
        stale.write_bytes(b"stale")
        result = assemble_object(plan, dataset, staging, "test-backup")
        self.assertEqual(result["status"], "ASSEMBLED")
        self.assertEqual(final.read_bytes(), b"ABCDE")
        backed_up = list((staging / "backup").rglob("*"))
        self.assertTrue(any(path.is_file() and path.read_bytes() == b"X" for path in backed_up))
        self.assertTrue(any(path.is_file() and path.read_bytes() == b"stale" for path in backed_up))

    def test_controlled_temp_partial_size_chunk_is_backed_up_then_redownloaded(self) -> None:
        _, _, staging = controlled_sandbox()
        plan = ObjectPlan(
            relative_path="sub-SD011/file.edf",
            source_url="https://s3.amazonaws.com/openneuro.org/ds004703/sub-SD011/file.edf",
            expected_bytes=5,
            modified_at_utc="2026-08-11T00:00:00Z",
            chunks=(RangeChunk(0, 4, 5),),
        )
        destination = chunk_path(staging, plan.relative_path, plan.chunks[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"X")
        response = FakeRangeResponse(b"ABCDE", "bytes 0-4/5")
        with patch("scripts.public_s3_range_download.urllib.request.urlopen", return_value=response):
            result = download_chunk(plan, plan.chunks[0], staging, "test-redownload")
        self.assertEqual(result["status"], "DOWNLOADED")
        self.assertEqual(destination.read_bytes(), b"ABCDE")
        backups = list((staging / "backup" / "chunk_size_mismatch").rglob("*.chunk"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"X")

    def test_controlled_temp_move_rejects_source_or_destination_outside_roots(self) -> None:
        root, dataset, staging = controlled_sandbox()
        outside = root / "outside.bin"
        outside.write_bytes(b"X")
        with self.assertRaises(ValueError):
            move_to_backup(
                outside,
                dataset,
                staging / "backup",
                staging,
                Path("outside.bin"),
            )
        inside = dataset / "inside.bin"
        inside.write_bytes(b"Y")
        with self.assertRaises(ValueError):
            move_to_backup(
                inside,
                dataset,
                root / "outside-backup",
                staging,
                Path("inside.bin"),
            )
        self.assertTrue(inside.exists())
        self.assertTrue(outside.exists())

    def test_zero_elapsed_rate_is_explicitly_not_estimated(self) -> None:
        self.assertIsNone(safe_mib_per_second(1024, 0.0))
        self.assertIsNone(safe_mib_per_second(1024, -1.0))
        self.assertEqual(safe_mib_per_second(1024 * 1024, 2.0), 0.5)

    def test_controlled_temp_exact_interrupted_set_moves_without_deleting(self) -> None:
        root, dataset, _ = controlled_sandbox()
        allowed_backup = root / "allowed-interrupted"
        backup = allowed_backup / "run-1"
        relative_paths = [
            "sub-SD011/file.edf.partial-one",
            "sub-SD019/file.edf.partial-two",
            "sub-SD022/file.edf.partial-three",
        ]
        for index, relative in enumerate(relative_paths, start=1):
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([index]) * index)
        preserved = dataset / "preserved.tsv"
        preserved.write_bytes(b"keep")
        report = move_interrupted_partials(dataset, backup, allowed_backup, relative_paths)
        self.assertEqual(report["moved_count"], 3)
        self.assertEqual(report["remaining_partial_count"], 0)
        self.assertEqual(len([path for path in backup.rglob("*") if path.is_file()]), 3)
        self.assertEqual(preserved.read_bytes(), b"keep")

    def test_controlled_temp_unexpected_partial_blocks_all_moves(self) -> None:
        root, dataset, _ = controlled_sandbox()
        allowed_backup = root / "allowed-interrupted"
        backup = allowed_backup / "run-2"
        expected = dataset / "sub-SD011/file.edf.partial-one"
        unexpected = dataset / "sub-SD019/file.edf.partial-unexpected"
        expected.parent.mkdir(parents=True, exist_ok=True)
        unexpected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_bytes(b"one")
        unexpected.write_bytes(b"two")
        with self.assertRaises(ValueError):
            move_interrupted_partials(
                dataset,
                backup,
                allowed_backup,
                ["sub-SD011/file.edf.partial-one"],
            )
        self.assertTrue(expected.exists())
        self.assertTrue(unexpected.exists())
        self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()

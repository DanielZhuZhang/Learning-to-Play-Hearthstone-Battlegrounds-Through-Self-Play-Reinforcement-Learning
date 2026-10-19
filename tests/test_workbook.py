from pathlib import Path

import pytest
from openpyxl import Workbook

from hsbg.data.workbook import EXPECTED_HEADERS, WorkbookDataError, load_sheet

WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "season -1 (beta).xlsx"


def _write_workbook(
    path: Path,
    rows: list[tuple[object, ...]],
    *,
    sheet_name: str = "15.6",
    headers: tuple[str, ...] = EXPECTED_HEADERS,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def _alleycat_row() -> tuple[object, ...]:
    return (
        1,
        1,
        "*",
        "Alleycat",
        "Beast",
        "Beast",
        1,
        1,
        "Battlecry: Summon a 1/1 Cat.",
    )


def test_loads_all_patch_15_6_rows_and_maps_attack_health_headers() -> None:
    cards = load_sheet(WORKBOOK_PATH, "15.6")

    assert len(cards) == 80
    alleycat = cards[0]
    assert alleycat.card_id == "15.6:1"
    assert alleycat.source.sheet == "15.6"
    assert alleycat.source.row == 2
    assert alleycat.name == "Alleycat"
    assert alleycat.attack == 1
    assert alleycat.health == 1
    assert alleycat.source_values == _alleycat_row()


def test_untyped_cards_are_normalized_but_raw_values_are_kept() -> None:
    cards = load_sheet(WORKBOOK_PATH, "15.6")
    protector = next(card for card in cards if card.name == "Righteous Protector")
    amalgam = next(card for card in cards if card.name == "Nightmare Amalgam")

    assert protector.tribe is None
    assert protector.card_type is None
    assert protector.source_values[4:6] == ("-", "-")
    assert amalgam.tribe is None
    assert amalgam.card_type == "All"
    assert amalgam.source_values[4:6] == ("-", "All")


def test_sheet_names_are_independent_card_id_namespaces() -> None:
    first = load_sheet(WORKBOOK_PATH, "15.6")[0]
    second = load_sheet(WORKBOOK_PATH, "15.6.2")[0]

    assert first.card_id == "15.6:1"
    assert second.card_id == "15.6.2:1"


def test_missing_workbook_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_sheet(tmp_path / "missing.xlsx", "15.6")


def test_missing_sheet_is_reported_with_available_sheets() -> None:
    with pytest.raises(WorkbookDataError, match="Available sheets: 15.6"):
        load_sheet(WORKBOOK_PATH, "unknown")


def test_missing_required_header_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "missing-header.xlsx"
    headers = tuple(header for header in EXPECTED_HEADERS if header != "⚔")
    row = tuple(value for index, value in enumerate(_alleycat_row()) if index != 6)
    _write_workbook(path, [row], headers=headers)

    with pytest.raises(WorkbookDataError, match="missing required columns: ⚔"):
        load_sheet(path, "15.6")


def test_duplicate_source_numbers_are_rejected_with_row_locations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.xlsx"
    _write_workbook(path, [_alleycat_row(), _alleycat_row()])

    with pytest.raises(WorkbookDataError, match="duplicate card ID '15.6:1'.*row 2"):
        load_sheet(path, "15.6")


@pytest.mark.parametrize(
    ("column_index", "value", "message"),
    [
        (0, None, "column '#' must be an integer"),
        (1, "one", "column 'Tier' must be an integer"),
        (1, 0, "tier must be positive"),
        (3, None, "column 'Name' must contain non-empty text"),
        (6, "sharp", "column '⚔' must be an integer"),
        (7, -1, "Base attack and health must not be negative"),
        (8, 42, "column 'Text' must contain text or empty"),
    ],
)
def test_invalid_source_values_report_sheet_and_row(
    tmp_path: Path,
    column_index: int,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "invalid-value.xlsx"
    row = list(_alleycat_row())
    row[column_index] = value
    _write_workbook(path, [tuple(row)])

    with pytest.raises(WorkbookDataError, match=message):
        load_sheet(path, "15.6")


def test_workbook_is_opened_read_only_and_never_modified() -> None:
    before = WORKBOOK_PATH.read_bytes()
    load_sheet(WORKBOOK_PATH, "15.6")
    after = WORKBOOK_PATH.read_bytes()

    assert after == before

"""Read-only import of versioned card data from the supplied Excel workbook."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from openpyxl import load_workbook  # type: ignore[import-untyped]
from openpyxl.worksheet.worksheet import Worksheet  # type: ignore[import-untyped]

from hsbg.data.models import CardRecord, CardSource, CellValue

EXPECTED_HEADERS = ("#", "Tier", "*s", "Name", "Tribe", "Type", "⚔", "✙", "Text")
_HEADER_INDEX = {header: index for index, header in enumerate(EXPECTED_HEADERS)}


class WorkbookDataError(ValueError):
    """Raised when a workbook sheet does not match the supported card schema."""


def load_sheet(workbook_path: str | Path, sheet_name: str) -> tuple[CardRecord, ...]:
    """Load a card sheet, preserving each source row while exposing named fields.

    The workbook is opened read-only and never modified. The sheet's `⚔` and
    `✙` columns map to the model's `attack` and `health` fields, respectively.
    """
    path = Path(workbook_path)
    if not path.is_file():
        raise FileNotFoundError(f"Card workbook does not exist: {path}")

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            available = ", ".join(workbook.sheetnames)
            raise WorkbookDataError(
                f"Sheet {sheet_name!r} not found in {path}. "
                f"Available sheets: {available}."
            )
        sheet = workbook[sheet_name]
        return _read_cards(sheet, path, sheet_name)
    finally:
        workbook.close()


def _read_cards(
    sheet: Worksheet, workbook_path: Path, sheet_name: str
) -> tuple[CardRecord, ...]:
    rows: Iterator[tuple[CellValue, ...]] = sheet.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None:
        raise WorkbookDataError(f"Sheet {sheet_name!r} is empty.")

    if tuple(header) != EXPECTED_HEADERS:
        missing = [name for name in EXPECTED_HEADERS if name not in header]
        if missing:
            raise WorkbookDataError(
                f"Sheet {sheet_name!r} is missing required columns: "
                f"{', '.join(missing)}."
            )
        raise WorkbookDataError(
            f"Sheet {sheet_name!r} has unexpected columns or order. "
            f"Expected {EXPECTED_HEADERS!r}; got {tuple(header)!r}."
        )

    cards: list[CardRecord] = []
    seen_ids: dict[str, int] = {}
    for excel_row, values in enumerate(rows, start=2):
        if all(value is None for value in values):
            continue
        if len(values) != len(EXPECTED_HEADERS):
            raise WorkbookDataError(
                f"Sheet {sheet_name!r}, row {excel_row}: expected "
                f"{len(EXPECTED_HEADERS)} cells, found {len(values)}."
            )

        source_number = _integer_cell(values, "#", sheet_name, excel_row)
        card_id = f"{sheet_name}:{source_number}"
        if card_id in seen_ids:
            raise WorkbookDataError(
                f"Sheet {sheet_name!r}, row {excel_row}: duplicate card ID "
                f"{card_id!r}; first seen on row {seen_ids[card_id]}."
            )

        tier = _integer_cell(values, "Tier", sheet_name, excel_row)
        attack = _integer_cell(values, "⚔", sheet_name, excel_row)
        health = _integer_cell(values, "✙", sheet_name, excel_row)
        name = _string_cell(values, "Name", sheet_name, excel_row, required=True)
        tribe = _optional_string_cell(values, "Tribe", sheet_name, excel_row)
        card_type = _optional_string_cell(values, "Type", sheet_name, excel_row)
        stars = _optional_string_cell(values, "*s", sheet_name, excel_row)
        text = _string_cell(values, "Text", sheet_name, excel_row, required=False)
        assert name is not None
        assert text is not None

        try:
            card = CardRecord(
                card_id=card_id,
                source=CardSource(
                    workbook=str(workbook_path),
                    sheet=sheet_name,
                    row=excel_row,
                ),
                source_number=source_number,
                tier=tier,
                stars=stars,
                name=name,
                tribe=tribe,
                card_type=card_type,
                attack=attack,
                health=health,
                text=text,
                source_values=values,
            )
        except ValueError as exc:
            raise WorkbookDataError(
                f"Sheet {sheet_name!r}, row {excel_row}: {exc}"
            ) from exc

        cards.append(card)
        seen_ids[card_id] = excel_row

    return tuple(cards)


def _integer_cell(
    values: tuple[CellValue, ...], header: str, sheet_name: str, row: int
) -> int:
    value = values[_HEADER_INDEX[header]]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkbookDataError(
            f"Sheet {sheet_name!r}, row {row}: column {header!r} "
            f"must be an integer; got {value!r}."
        )
    if isinstance(value, float) and not value.is_integer():
        raise WorkbookDataError(
            f"Sheet {sheet_name!r}, row {row}: column {header!r} "
            f"must be an integer; got {value!r}."
        )
    return int(value)


def _string_cell(
    values: tuple[CellValue, ...],
    header: str,
    sheet_name: str,
    row: int,
    *,
    required: bool,
) -> str | None:
    value = values[_HEADER_INDEX[header]]
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        requirement = "non-empty text" if required else "text or empty"
        raise WorkbookDataError(
            f"Sheet {sheet_name!r}, row {row}: column {header!r} "
            f"must contain {requirement}; got {value!r}."
        )
    return value


def _optional_string_cell(
    values: tuple[CellValue, ...], header: str, sheet_name: str, row: int
) -> str | None:
    value = values[_HEADER_INDEX[header]]
    if value is None or value == "" or value == "-":
        return None
    if not isinstance(value, str):
        raise WorkbookDataError(
            f"Sheet {sheet_name!r}, row {row}: column {header!r} "
            f"must contain text, '-', or an empty cell; got {value!r}."
        )
    return value

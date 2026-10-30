# Panel layout shared by every figure of the Copernicus plankton groups

# One column per size class, from the largest cells on the left to the smallest on the right. Each column opens with its size class and continues with the groups the product sums into it. Nanophytoplankton equals haptophytes on every row of the plankton gold table, so the nano column is the haptophyte panel alone
SIZE_CLASS_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('MICRO', ('DIATO', 'DINO')),
    ('HAPTO', ()),
    ('PICO', ('GREEN', 'PROKAR', 'PROCHLO')),
)
COPERNICUS_LABELS = {
    'MICRO': 'Microphytoplankton', 'DIATO': 'Diatoms', 'DINO': 'Dinophytes',
    'HAPTO': 'Nanophytoplankton (haptophytes)',
    'PICO': 'Picophytoplankton', 'GREEN': 'Green algae', 'PROKAR': 'Prokaryotes', 'PROCHLO': 'Prochlorophytes',
}
# Column by column, so panel letters run down each size class
SIZE_CLASS_CELLS: list[tuple[int, int, str]] = [
    (row, column, field)
    for column, (size_class, members) in enumerate(SIZE_CLASS_COLUMNS)
    for row, field in enumerate((size_class, *members))
]
SIZE_CLASS_FIELDS = [field for _, _, field in SIZE_CLASS_CELLS]
SIZE_CLASS_GRID_SHAPE = (max(row for row, _, _ in SIZE_CLASS_CELLS) + 1, len(SIZE_CLASS_COLUMNS))


# The row of the lowest panel in each column, which carries that column's x tick labels
def find_lowest_rows(cells: list[tuple[int, int, str]]) -> dict[int, int]:
    lowest: dict[int, int] = {}
    for row, column, _ in cells:
        lowest[column] = max(row, lowest.get(column, row))
    return lowest


# Grid positions without a panel, in reading order; the first one holds the legend
def find_spare_cells(cells: list[tuple[int, int, str]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    used = {(row, column) for row, column, _ in cells}
    return [(row, column) for row in range(shape[0]) for column in range(shape[1]) if (row, column) not in used]

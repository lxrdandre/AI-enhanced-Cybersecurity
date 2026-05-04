/**
 * Normalize text for case-insensitive page filtering.
 */
function pageEscape(value) {
  return String(value ?? "").toLowerCase();
}

/**
 * Apply table filters to the current page.
 */
function applyTableFilters(tableId) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const controls = Array.from(document.querySelectorAll(`[data-filter-table="${tableId}"]`));
  const rows = Array.from(table.querySelectorAll("tbody tr"));
  rows.forEach(row => {
    const cells = Array.from(row.children).map(cell => pageEscape(cell.textContent));
    const visible = controls.every(control => {
      const value = pageEscape(control.value).trim();
      if (!value) return true;
      const column = control.dataset.filterColumn;
      if (column !== undefined) {
        const text = cells[Number(column)] || "";
        return text.includes(value);
      }
      return cells.some(text => text.includes(value));
    });
    row.hidden = !visible;
  });
}

document.querySelectorAll("[data-filter-table]").forEach(control => {
  control.addEventListener("input", () => applyTableFilters(control.dataset.filterTable));
  control.addEventListener("change", () => applyTableFilters(control.dataset.filterTable));
});

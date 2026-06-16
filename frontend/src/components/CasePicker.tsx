import { useMemo, useState } from "react";

interface CasePickerProps {
  cases: string[];
  selected: string | null;
  onSelect: (id: string) => void;
  loading?: boolean;
  error?: string | null;
}

/** Sidebar: searchable list of the 103 benchmark cases. */
export function CasePicker({ cases, selected, onSelect, loading, error }: CasePickerProps) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return cases;
    return cases.filter((c) => c.toLowerCase().includes(q));
  }, [cases, query]);

  return (
    <div className="case-picker">
      <div className="case-picker__head">
        <label className="case-picker__title">Cases · 用例</label>
        <input
          className="case-picker__search"
          type="search"
          placeholder="搜索 / search (e.g. t001)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search cases"
        />
        <span className="case-picker__count">
          {loading ? "loading…" : `${filtered.length} / ${cases.length}`}
        </span>
      </div>

      {error && <div className="case-picker__error">Failed to load cases: {error}</div>}

      <ul className="case-picker__list">
        {filtered.map((c) => (
          <li key={c}>
            <button
              type="button"
              className={`case-picker__item ${selected === c ? "case-picker__item--active" : ""}`}
              onClick={() => onSelect(c)}
            >
              <span className="case-picker__id">{c}</span>
            </button>
          </li>
        ))}
        {!loading && filtered.length === 0 && (
          <li className="case-picker__empty">No matching cases.</li>
        )}
      </ul>
    </div>
  );
}

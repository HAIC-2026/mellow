import { useMemo, useState, type FormEvent } from "react";
import {
  getGraph,
  searchEntities,
  type Citation,
  type EntityMatch,
  type GraphEdge,
  type GraphResponse,
} from "../api/client";
import GraphCanvas, { type CanvasEdgeData } from "../components/GraphCanvas";
import SourceViewer from "./SourceViewer";

const MAX_NODES = 150;

function edgeKey(e: GraphEdge): string {
  return `${e.src}-${e.dst}-${e.type}`;
}

export default function GraphPage() {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<EntityMatch[] | null>(null);
  const [centre, setCentre] = useState<number | null>(null);
  const [depth, setDepth] = useState(2);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [citations, setCitations] = useState<Citation[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function search(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      setMatches(await searchEntities(query));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function load(entity: number, d = depth) {
    setError(null);
    setSelectedKey(null);
    setCitations(null);
    try {
      setCentre(entity);
      setGraph(await getGraph(entity, d, MAX_NODES));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function changeDepth(d: number) {
    setDepth(d);
    if (centre !== null) load(centre, d);
  }

  const edgesByKey = useMemo(() => {
    const map = new Map<string, GraphEdge>();
    for (const e of graph?.edges ?? []) map.set(edgeKey(e), e);
    return map;
  }, [graph]);

  const canvasNodes = useMemo(
    () =>
      (graph?.nodes ?? []).map((n) => ({
        id: n.id,
        name: n.name,
        type: n.type,
        size: Math.min(60, 14 + 4 * n.mentions),
      })),
    [graph],
  );

  const canvasEdges: CanvasEdgeData[] = useMemo(
    () =>
      (graph?.edges ?? []).map((e) => ({
        key: edgeKey(e),
        src: e.src,
        dst: e.dst,
        label: e.type,
        causal: e.class === "causal",
        negative: e.assertions.every((a) => a.polarity === "negative"),
        conflict: e.conflict,
        review: e.review,
        width: 1 + Math.min(e.documents, 5),
      })),
    [graph],
  );

  const selected = selectedKey ? edgesByKey.get(selectedKey) ?? null : null;
  const nameOf = (id: number) => graph?.nodes.find((n) => n.id === id)?.name ?? String(id);

  return (
    <div>
      <h2>Graph</h2>
      <form onSubmit={search} style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Entity name or alias"
        />
        <button type="submit">Search</button>
        <label>
          Depth{" "}
          <select value={depth} onChange={(e) => changeDepth(Number(e.target.value))}>
            {[1, 2, 3, 4].map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
      </form>
      {error && <p role="alert">{error}</p>}
      {matches && (
        <ul>
          {matches.length === 0 && <li>No entity matches.</li>}
          {matches.map((m) => (
            <li key={m.id}>
              <button onClick={() => load(m.id)}>{m.name}</button> ({m.type}, {m.mentions} mentions)
            </li>
          ))}
        </ul>
      )}
      {graph && (
        <p>
          {graph.nodes.length} entities, {graph.edges.length} links.
          {graph.truncated && ` Stopped at ${MAX_NODES} entities; lower the depth to see all.`}
          {" "}Double-click a node to centre on it. Click a link to list its assertions.
        </p>
      )}
      {graph && (
        <div style={{ display: "flex", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <GraphCanvas
              nodes={canvasNodes}
              edges={canvasEdges}
              centreId={centre ?? undefined}
              onNodeDoubleTap={(id) => load(id)}
              onEdgeTap={(key) => setSelectedKey(key)}
            />
          </div>
          {selected && (
            <aside style={{ width: 360 }}>
              <h3>
                {nameOf(selected.src)} → {nameOf(selected.dst)}
              </h3>
              <p>
                {selected.type} ({selected.class})
                {selected.conflict && " · documents disagree"}
                {selected.review && " · causal in both directions: review"}
              </p>
              <ul>
                {selected.assertions.map((a) => (
                  <li key={a.id}>
                    <strong>{a.predicate}</strong> [{a.polarity}, {a.explicit ? "explicit" : "inferred"}] {a.doc_name}{" "}
                    <button onClick={() => setCitations(a.spans)}>Open source</button>
                  </li>
                ))}
              </ul>
            </aside>
          )}
        </div>
      )}
      {citations && <div style={{ marginTop: 12 }}><SourceViewer key={citations.map((c) => c.span_id).join(",")} citations={citations} /></div>}
    </div>
  );
}

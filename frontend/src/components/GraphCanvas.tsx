import { useEffect, useRef } from "react";
import cytoscape, { type Core, type ElementDefinition } from "cytoscape";

export interface CanvasEdgeData {
  // Edge identity, passed back to onEdgeTap.
  key: string;
  src: number;
  dst: number;
  label: string;
  causal: boolean;
  negative: boolean;
  conflict: boolean;
  review: boolean;
  width: number;
}

interface Props {
  nodes: { id: number; name: string; type: string; size: number }[];
  edges: CanvasEdgeData[];
  centreId?: number;
  onNodeDoubleTap?: (id: number) => void;
  onEdgeTap?: (key: string) => void;
}

// One colour per entity type, stable across renders.
function colourFor(type: string): string {
  let hash = 0;
  for (const ch of type) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsl(${hash % 360} 55% 62%)`;
}

export default function GraphCanvas({ nodes, edges, centreId, onNodeDoubleTap, onEdgeTap }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const handlers = useRef({ onNodeDoubleTap, onEdgeTap });
  handlers.current = { onNodeDoubleTap, onEdgeTap };

  useEffect(() => {
    if (!container.current) return;
    const instance = cytoscape({
      container: container.current,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-size": 11,
            "text-wrap": "wrap",
            "text-max-width": "120px",
            "background-color": "data(colour)",
            width: "data(size)",
            height: "data(size)",
            "border-width": 0,
          },
        },
        { selector: "node.centre", style: { "border-width": 3, "border-color": "#222" } },
        {
          selector: "edge",
          style: {
            width: "data(width)",
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#555",
            "target-arrow-color": "#555",
            label: "data(label)",
            "font-size": 9,
            "text-rotation": "autorotate",
          },
        },
        { selector: "edge[?dashed]", style: { "line-style": "dashed" } },
        { selector: "edge[?negative]", style: { "line-color": "#c0392b", "target-arrow-color": "#c0392b" } },
        { selector: "edge[?conflict]", style: { "border-width": 2, "border-color": "#c0392b", "border-style": "solid" } },
        { selector: "edge[?review]", style: { "line-style": "dotted" } },
      ],
      layout: { name: "cose", animate: false },
      minZoom: 0.2,
      maxZoom: 3,
    });
    instance.on("dbltap", "node", (evt) => handlers.current.onNodeDoubleTap?.(evt.target.data("entityId")));
    instance.on("tap", "edge", (evt) => handlers.current.onEdgeTap?.(evt.target.data("key")));
    cy.current = instance;
    return () => {
      instance.destroy();
      cy.current = null;
    };
  }, []);

  useEffect(() => {
    const instance = cy.current;
    if (!instance) return;
    const elements: ElementDefinition[] = [
      ...nodes.map((n) => ({
        group: "nodes" as const,
        data: { id: `n${n.id}`, entityId: n.id, label: n.name, colour: colourFor(n.type), size: n.size },
        classes: n.id === centreId ? "centre" : "",
      })),
      ...edges.map((e) => ({
        group: "edges" as const,
        data: {
          id: e.key,
          key: e.key,
          source: `n${e.src}`,
          target: `n${e.dst}`,
          label: e.label + (e.review ? " (review)" : ""),
          width: e.width,
          dashed: !e.causal,
          negative: e.negative,
          conflict: e.conflict,
          review: e.review,
        },
      })),
    ];
    instance.elements().remove();
    instance.add(elements);
    instance.layout({ name: "cose", animate: false, fit: true, padding: 20 }).run();
  }, [nodes, edges, centreId]);

  return <div ref={container} className="graph-canvas" style={{ height: 480, border: "1px solid #ddd" }} />;
}

from typing import Dict

from api import NERAPI
from dag_module import DAGBuilder, T2Entity


def process_tee_request(tee_request: Dict) -> Dict:
    t2_entities = tee_request.get("t2_entities", [])
    epsilon = float(tee_request.get("epsilon", 1.0))

    _entities, raw_edges = NERAPI()(tee_request.get("fpe_sanitized_text", ""))

    entities = [
        T2Entity(
            index=idx,
            token=entity.get("token", ""),
            value=entity.get("value", 0),
        )
        for idx, entity in enumerate(t2_entities)
    ]

    graphs = DAGBuilder().build(entities, raw_edges, epsilon)

    relation_edges = []
    for graph in graphs:
        for edge in graph.edges:
            relation_edges.append({
                "from_entity_index": edge.parent_idx,
                "to_entity_index": edge.child_idx,
                "relation_type": edge.rel_type.value,
                "param": edge.param,
                "has_temp_node": edge.child_idx < 0,
                "temp_node_name": "",
            })

    return {
        "edges": relation_edges,
        "attestation_report": "",
    }

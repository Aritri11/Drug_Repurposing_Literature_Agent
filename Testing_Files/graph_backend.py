# graph_backend.py

from neo4j import GraphDatabase


class Neo4jGraph:
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password

    def get_driver(self):
        return GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            max_connection_lifetime=3600,
            connection_timeout=30,
            max_connection_pool_size=50
        )





# class Neo4jGraph:
#     def __init__(self, uri, username, password):
#         self.driver = GraphDatabase.driver(uri, auth=(username, password))
#
#     def close(self):
#         self.driver.close()
#
#     def get_ranked_candidates(self, disease_name: str):
#         query = """
#         MATCH (d:Drug)-[m:MODULATES]->(g:Gene)
#         MATCH (di:Disease {name:$disease})-[e:ALTERS_EXPRESSION]->(g)
#
#         WHERE
#             (m.action = "ACTIVATES" AND e.direction = "DOWN")
#             OR
#             (m.action = "INHIBITS" AND e.direction = "UP")
#
#         WITH d, g,
#              (m.confidence_score * e.confidence_score *
#               log(1 + m.evidence_count) *
#               log(1 + e.evidence_count)) AS match_score
#
#         RETURN d.name AS CandidateDrug,
#                COLLECT(g.name) AS SupportingGenes,
#                SUM(match_score) AS TotalScore
#         ORDER BY TotalScore DESC
#         """
#
#         with self.driver.session() as session:
#             result = session.run(query, disease=disease_name)
#
#             candidates = []
#             for record in result:
#                 candidates.append({
#                     "drug": record["CandidateDrug"],
#                     "supporting_genes": record["SupportingGenes"],
#                     "score": record["TotalScore"]
#                 })
#
#             return candidates

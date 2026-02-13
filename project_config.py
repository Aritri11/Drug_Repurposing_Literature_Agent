import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI="neo4j+s://91da1fda.databases.neo4j.io"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="CDEenHZbTMIKTm9p7I1S0Slb4V5u_0zOPfCcxDd3Kj0"
EMAIL="aritribaidya8@gmail.com"





#test connection
# from neo4j import GraphDatabase
#
# driver = GraphDatabase.driver(
#     "neo4j+s://91da1fda.databases.neo4j.io",
#     auth=("neo4j","CDEenHZbTMIKTm9p7I1S0Slb4V5u_0zOPfCcxDd3Kj0")
# )
#
# with driver.session() as s:
#     print(s.run("RETURN 'Connected!'").single())

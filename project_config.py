import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI="YOUR URI ID"
NEO4J_USER="USER NAME"
NEO4J_PASSWORD="YOUR PASSWORD"
EMAIL="YOURMAILID@gmail.com"





#test connection
# from neo4j import GraphDatabase
#
# driver = GraphDatabase.driver(
#     "YOUR URI ID",
#     auth=("USER NAME","YOUR PASSWORD")
# )
#
# with driver.session() as s:
#     print(s.run("RETURN 'Connected!'").single())


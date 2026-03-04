#This code hits the DGIdb API and fetches the drugs associated with a gene along with the interaction type
import requests
import json


def fetch_dgidb_interactions(gene_symbol: str):
    url = "https://dgidb.org/api/graphql"

    # Write the GraphQL query. You specify exactly what data you want back.
    query = """
    query GetInteractions($geneName: [String!]) {
      genes(names: $geneName) {
        nodes {
          name
          interactions {
            drug {
              name
              approved
            }
            interactionScore
            interactionTypes {
              type
              directionality
            }
            publications {
              pmid
            }
          }
        }
      }
    }
    """

    # Pass the gene symbol as a variable
    variables = {
        "geneName": [gene_symbol]
    }

    # Make the POST request
    response = requests.post(url, json={'query': query, 'variables': variables})

    if response.status_code == 200:
        data = response.json()

        # Safely extract the interactions list
        try:
            interactions = data['data']['genes']['nodes'][0]['interactions']
            return interactions
        except (KeyError, IndexError):
            return []
    else:
        print(f"Query failed with status code {response.status_code}")
        return []


# # --- Test the function ---
# if __name__ == "__main__":
#     gene = "AKT2"
#     print(f"🔎 Fetching drugs that target {gene} from DGIdb...")
#
#     interactions = fetch_dgidb_interactions(gene)
#
#     print(f"Found {len(interactions)} interactions. Here are the top 3:")
#     print(json.dumps(interactions[:3], indent=2))
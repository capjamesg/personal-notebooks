import os

import pandas as pd

import src.pyatproto as atproto

ENDPOINT = os.environ.get("ATPROTO_ENDPOINT")
USERNAME = os.environ.get("ATPROTO_USERNAME")
PASSWORD = os.environ.get("ATPROTO_PASSWORD")

if not ENDPOINT or not USERNAME or not PASSWORD:
    raise ValueError("Please set the ATPROTO_ENDPOINT, ATPROTO_USERNAME and ATPROTO_PASSWORD environment variables.")

ap = atproto.AtProtoConfiguration(ENDPOINT, USERNAME, PASSWORD)

my_did = ap.did

following = ap.get_following(my_did)

second_degree_connections = []

for user in following["records"]:
    uri = user["value"]["subject"]

    for second_degree in ap.get_following(uri)["records"]:
        print(second_degree)
        second_degree_connections.append(
            {
                "userFollowing": uri,
                "userBeingFollowed": second_degree["value"]["subject"]
            }
        )

df = pd.DataFrame(second_degree_connections)

df.to_csv("second_degree_connections.csv")
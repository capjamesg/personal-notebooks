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

posts = []

limit = 0

try:
    for follow in following["records"]:
        # print(follow)
        uri = follow["value"]["subject"]

        # get likes from the user
        likes = ap.get_likes(uri)

        for like in likes["records"][:25]:
            post = ap.get_post(like["value"]["subject"]["uri"])

            print(limit)

            has_liked = False

            if post["thread"]["post"]["author"]["did"] == my_did:
                has_liked = True

            posts.append(
                {
                    "text": post["thread"]["post"]["record"]["text"],
                    "likes": post["thread"]["post"]["likeCount"],
                    "reposts": post["thread"]["post"]["repostCount"],
                    "createdAt": post["thread"]["post"]["record"]["createdAt"],
                    "uri": post["thread"]["post"]["uri"],
                    "user": uri,
                    "hasLiked": has_liked
                }
            )
            print(uri)
            limit += 1

        if limit > 1000:
            break
except Exception as e:
    print(e)

print(posts)

df = pd.DataFrame(posts, columns=["text", "likes", "reposts", "createdAt", "uri", "user", "hasLiked"])

# save to csv
df.to_csv("likes.csv", index=False)

# ap.get_likes(did)
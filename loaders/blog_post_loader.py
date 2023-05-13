# load all text files in /Users/james/src/jamesg.blog/_posts, get text from frontmatter

import os
import frontmatter

directory = "/Users/james/src/jamesg.blog/_posts"
contents = []
categories = []

for filename in os.listdir(directory):
    if filename.endswith(".md"):
        with open(os.path.join(directory, filename)) as f:
            post = frontmatter.load(f)
            contents.append(post.content)
            categories.append(post['categories'][0])
    else:
        continue

# save to csv, alongside category
import pandas as pd
df = pd.DataFrame({'content': contents, 'category': categories})
df.to_csv('posts.csv', index=False)
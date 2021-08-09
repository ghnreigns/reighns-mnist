- [10. `esc + q` to escape menu.](#10-esc--q-to-escape-menu)
  - [gitignore](#gitignore)
  - [Folder Structure](#folder-structure)

### Starting out
https://madewithml.com/courses/mlops/git/

1. Open the folder in vscode and type `git init` in terminal. For windows user, you should install git properly first.
2. Create a file in the folder - for linux you can use `touch main.py` for windows you just use `code filename.py`.
3. git add the file by using `git add main.py`
4. `git status` to check status of tracked vs untracked files.
5. `git commit -a` and write your message, or `git commit -a main.py`, note if you are using vim, you need to press insert, and type your message, subsequenty press esc and type :wq to commit, or :qa to exit.
6. Note everything done above is still at local repository.
7. `git remote add origin https://github.com/reigHns/reighns-mnist.git` so that you have added this local repo to the github or bitbucket. Note to remove just use `git remote remove origin to remove`.
8. `git push origin master` to push to master branch - merge branch we do next time.
    1. One thing worth noting is that if you created the repository with some files in it, then the above steps will lead to error.
    2. Instead, to overwrite : do `git push -f origin master` . For newbies, it is safe to just init a repo with no files inside in the remote to avoid the error "Updates were rejected because the tip..."
9. Now step 8 may fail sometimes if you did not verify credentials, a fix is instead of step 7 and 8, we replace with 
    ```
    git remote add origin your-repo-http
    git remote set-url origin https://reighns@github.com/reighns/reighns-mnist.git
    git push origin master
    ```
10. `esc + q` to escape menu.
---

### gitignore

1. I have secret keys for s3 bucket and do not want to commit to repository.
2. Create a `.gitignore` file and `keys.py` where you keep the secret keys in the latter, and add this file name into `.gitignore` such that when commiting and pushing, these won't be pushed to github.
3. `git add `.gitignore` and push it to server, and note when you check `git status` the untracked files do not consist of `keys.py`.

---

### Folder Structure

This issue should be solved properly with command line `args`. Maybe?

Project

MIDAS

deployment folder

deploy_comments.py

main.py

keys.py

To import keys in deployment folder's comment script. Do the following and remember to add `__init__`files.

```python
import sys
import os

sys.path.append(os.getcwd())
from MIDAS import keys
```

However, consider the idea that we can create a `deploy_main.py` file in the parent directory.

1. If you write some code and want to push again. we do the following
    - git commit -a and write your comments
    - git push origin master
2. use `git log -p` to find changes differences etc and press Q to exit.
3. `git status`: 3 categories to see your files that are tracked, commited and untracked. Untracked is files you have not added. If you have files in gitignore, it won't show in status.
4. `git checkout`: followed by the unique (commit sha), it will revert your code back to that version. And `git checkout master (or the branch)` to get back to current.
5. Use `git diff` to check out differences between versions.
6. More importantly, when you create new files, need to `git add` so they are tracked. Then `git commit and push`.

Here will push it to github or bitbucket.
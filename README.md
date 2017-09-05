# aiclass
aiclass
2017.09.02

# basic git command
- git init
- git config --global user.name "your name"
- git config --global user.email "your email"

- git fetch origin master
- git pull origin master

- git commit -m "message"
- git add *
- git push origin master

# how to push without ID/PW

$ git config credential.helper store
$ git push https://github.com/repo.git

Username for 'https://github.com': <USERNAME>
Password for 'https://USERNAME@github.com': <PASSWORD>

git config --global credential.helper 'cache --timeout 7200'


source from: https://stackoverflow.com/questions/6565357/git-push-requires-user$




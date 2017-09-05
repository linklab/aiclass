# aiclass
aiclass
2017.09.02

## basic git command
- git init
- git config --global user.name "your name"
- git config --global user.email "your email"
<br>
- git fetch origin master
- git pull origin master
<br>
- git commit -m "message"
- git add *
- git push origin master

## how to push without ID/PW

$ git config credential.helper store
$ git push https://github.com/repo.git

Username for 'https://github.com': <USERNAME><br>
Password for 'https://USERNAME@github.com': <PASSWORD><br><br>

git config --global credential.helper 'cache --timeout 7200'<br><br>


source from: https://stackoverflow.com/questions/6565357/git-push-requires-user$




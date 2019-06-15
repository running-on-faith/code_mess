echo %date% >> changes.log
git status
git status >> changes.log
git diff >> changes.log
git add .
git commit -m "update files"
git push

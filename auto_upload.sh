echo %date% >> changes.log
git status
echo ">>git status" >> changes.log
git status >> changes.log
echo ">>git diff" >> changes.log
git diff >> changes.log
git add .
git commit -m "update files"
git push

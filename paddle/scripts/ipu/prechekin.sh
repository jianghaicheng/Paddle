cd /paddle/python
source /popsdk/poplar-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
source /popsdk/popart-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
export PYTHONPATH=/paddle/build/python:$PYTHONPATH
test_dir=/paddle/python/paddle/fluid/tests/unittests/ipu/
for file in `ls $test_dir | grep -i test`
do
 echo "$file"
 echo $test_dir/$file
 if [ "${file##*.}"x = "py"x ]; then
   pytest $test_dir/$file -s > log 2>&1
 fi
done
result1=`grep "failed" log`
result2=`grep "ERRORS" log`
cat ./log
if [ -n "$result1" ] || [ -n "$result2" ]; then
echo "Test Failed"
exit 1
else
echo "Test Passs"
fi

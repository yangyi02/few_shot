#/bin/bash
cd tests

for shfile in *.sh
do
  sh $shfile
done

cd ..

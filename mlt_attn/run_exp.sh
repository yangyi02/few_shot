#/bin/bash
cd exps

for shfile in *.sh
do
  sh $shfile
done

cd ..

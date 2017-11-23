rm -rf faces64
rm -rf faces-rnd64
mkdir faces64
mkdir faces-rnd64
for line in `cat blacklist2.txt`
do
  rm faces/$line
done
for line in `cat blacklist-rnd.txt`
do
  rm faces-rnd/$line
done
python resize.py faces
python resize.py faces-rnd

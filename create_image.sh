rm -rf faces
rm -rf faces64
rm -rf faces64_edge
rm -rf faces65_gray
rm -rf faces-np
rm -rf faces-np64
rm -rf faces-np64_edge
rm -rf faces-np64_gray
mkdir faces
mkdir faces64
mkdir faces64_edge
mkdir faces65_gray
mkdir faces-np
mkdir faces-np64
mkdir faces-np64_edge
mkdir faces-np64_gray
python get_face.py patio faces
for line in `cat blacklist.txt`
do
  rm faces/$line
done
python get_face.py non-patio faces-np
python resize.py faces
python resize.py faces-np

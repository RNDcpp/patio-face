rm -f faces
rm -f faces64
rm -f faces64_edge
rm -f faces65_gray
rm -f faces-np
rm -f faces-np64
rm -f faces-np64_edge
rm -f faces-np64_gray
mkdir faces
mkdir faces64
mkdir faces64_edge
mkdir faces65_gray
mkdir faces-np
mkdir faces-np64
mkdir faces-np64_edge
mkdir faces-np64_gray
python get_face.py patio faces
python get_face.py non-patio faces-np
python resize.py faces
python resize.py faces-np

swig -c++ -python accelerated_sequence_clustering.i
python setup.py build_ext
python setup.py install --install-platlib=.
python setup.py install
python test_sequence_clustering.py

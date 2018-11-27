all:
	python setup.py build_ext --inplace
	rm -rf build
	bash make.sh
clean:
	rm ./utils/*.so
	rm ./roi_pooling_layer/*.so
	rm ./nms/*.so

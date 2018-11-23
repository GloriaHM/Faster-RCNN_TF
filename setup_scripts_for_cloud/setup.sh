echo "Running setup"
pip install names
pip install ipython==5.5
pip install prettytable
pip install scipy
pip install pandas==0.19.2
pip install matplotlib

pip install easydict
pip install cython 
pip install opencv-python
pip install pillow
apt-get install libgtk2.0-dev -y
apt-get install python-tk -y

#If use prebuild env with tf support do not need the following
#pip install tensorflow==1.10.1
#pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp27-none-linux_x86_64.whl
#pip install -U protobuf

#download data 
mv VOCdevkit Faster-RCNN_TF/data/VOCdevkit2007

#download model
unzip models.zip
mv VGG_imagenet.npy Faster-RCNN_TF/data/pretrain_model/.
mv VGGnet_fast_rcnn_iter_70000.ckpt Faster-RCNN_TF/data/pretrain_model/.

#modify env
export PATH=$PATH:/usr/local/cuda/bin
sleep 1;
echo 'backend : Agg' > /root/.config/matplotlib/matplotlibrc
sleep 1;

#make necessary bianry
sleep 3;
cd Faster-RCNN_TF
sleep 1;
cd lib
make
cd ..

echo "Setup step done"

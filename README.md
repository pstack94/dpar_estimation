# dpar_estimation

## Generating training data

To generate data for the artificial neural network run: 

python create_data.py dp_min dp_max dfree bvalfilename bvecfile name f_out shells snr

were: 
* *dp_min* is de minimum parallel difusivity i.e 0.5e-3
* *dp_max* is de maximum parallel difusivity i.e 1e-3
* *dfree* is de free water difusivity  i.e 1.44e-3
* *bvalfilename* path an name of the bval file i.e path/file.bval
* *bvecfilename* path an name of the bvec file i.e path/file.bvec
* *f_out path* where the data is going to be stored i.e outputpath/
* *shells* shells of the pretocol i.e 0 500 1500 
* *snr* snr of the training data i.e 22

Then upload the file data.zip to your google drive. 

## Training the ANN
Run the following googlecolab to train the ANN: [GoogleColab Notebook to train ANN](https://colab.research.google.com/drive/1HV0k8xS-tnIDxbd4ag34kqv4M6i5bTnG?usp=sharing).

save the .h5 file in your computer. 

#include "tensor.hpp"
#include <fstream> //std::ifstream
#include <iostream>  // std::ifstream
#include <algorithm> // std::max_element()

Tensor * load_record( std::ifstream & fin,int32_t n_dims, std::string name  ) {
  int32_t d[2] = { 1, 1 }; // number of element, nb=number of byte
  for (int i = 0; i < n_dims; ++i) // FC dimensions taken from file, eg. 768x500
    fin.read(reinterpret_cast<char *>(&d[i]), sizeof(d[i]) );
  int n_bytes = d[0] * d[1] * sizeof(float);
  float * buf = (float*) malloc(n_bytes );
  fin.read(reinterpret_cast<char *>( buf ), n_bytes );
  return new Tensor( d[0], d[1], buf , name );
}

bool mnist_model_load(const std::string & fname, Tensor * inp  ) { // load the model's weights from a file
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }
    uint32_t magic; // verify magic
    fin.read((char *) &magic, sizeof(magic));
    if (magic != 0x67676d6c) {
        fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
        return false;
    }
    int32_t n_dims; // Read FC1 layer 1, // Read dimensions
    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
	Tensor *fc1w = load_record( fin, n_dims,  "fc1w" );           inp->print();
	Tensor fc1    =  Tensor::matmul( *inp, *fc1w );  	          fc1.print();
	Tensor *fc1b  = load_record( fin, n_dims, "fc1b" );           fc1b->print();
	Tensor fc1bwa = *fc1b + fc1;
	Tensor fc1bw  = fc1bwa.relu(); fc1bw.print();
    // Read FC2 layer 2, Read dimensions
    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims)); // FC2 dimensions taken from file, eg. 10x500
    Tensor *fc2w  = load_record( fin, n_dims, "fc2w" );
    Tensor *fc2b  = load_record( fin, n_dims, "fc2b" );
    Tensor fc2r   = Tensor::matmul( fc1bw, *fc2w  );
    Tensor final1 = (fc2r + *fc2b).soft_max() ;

	float* finalData = final1.data();
	int prediction = std::max_element(finalData, finalData + 10) - finalData;
	std::cout << "Prediction : " << prediction << "\n";

    fin.close();
    return true;
}

Tensor * load_file() {
	std::vector<float> digit;
    auto fin = std::ifstream("./t10k-images.idx3-ubyte", std::ios::binary);
    if (!fin)  fprintf(stderr, "%s: failed to open '%s'\n", __func__, "t10k-images.idx3-ubyte");
    unsigned char buf[784];
    srand(time(NULL));
    // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
    int irand = rand();   //
    irand = 1100395855; // make it constant for debug
    fin.seekg(16 + 784 * ( irand % 10000));
    fin.read((char *) &buf, sizeof(buf));
    digit.resize(sizeof(buf));
    for(int row = 0; row < 28; row++) { // render the digit in ASCII
        for (int col = 0; col < 28; col++) {
            fprintf(stderr, "%c ", (float)buf[row*28 + col] > 177 ? 219  : '.'); // '#'
            digit[row*28+col]=((float)buf[row*28+col]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    return new Tensor(1,784, digit.data() );
}

int main() {
  Tensor * inp = load_file();	
  mnist_model_load( "./ggml-mnist-f32.bin",  inp  );
  return 0;
}


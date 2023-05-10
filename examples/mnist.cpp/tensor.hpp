#include <iostream> //std::cout
#include <iomanip>
#include <sstream>
#include <random> // log(), expf()
#include <cassert> // assert()
#include <cstring>  // std::memcpy

#define DEBUG 1

class Tensor {
  public:
    Tensor(unsigned int rows, unsigned int cols, float* data = nullptr, std::string name ="");
    Tensor copy();
    unsigned int cols() const;
    unsigned int rows() const;
    float* data() const;
    Tensor soft_max();
    Tensor relu() ;
    std::string toString();
    void print();

    static Tensor matmul(const Tensor &A, const Tensor &B);

    // Operators
    Tensor operator+(const Tensor &other);
    Tensor operator+(const float c);
    Tensor operator*(const Tensor &other);
    Tensor operator*(const float c);

    float& operator[](unsigned int index);
    float& operator()(unsigned int x, unsigned int y) const;

    std::string name ;

  private:
    // Internal input check
    static void checkEqualSize(const Tensor &A, const Tensor &B);
    static void checkMatMulPossible(const Tensor &A, const Tensor &B);
    static void checkInBounds(const Tensor &A, unsigned int x, unsigned int y);
    
    float *m_data;
    unsigned int m_rows;
    unsigned int m_cols;

};


Tensor::Tensor(unsigned int rows, unsigned int cols, float* data, std::string name ){
    m_data = new float[rows*cols]();
    this->name = name ;
    if(data){
        if ( rows <= cols )
          memcpy(m_data,data,rows*cols*sizeof(float));
        else {
            if ( DEBUG != 0 ) printf( "\tDo other copy rows=%d cols=%d\n", rows, cols );
            int c = 0;
            for (int i = 0; i < rows; i++) 
              for (int j = 0; j < cols; j++) m_data[c++] = data[j*rows+i];
        }
    }
    m_cols = cols;
    m_rows = rows;
    if (DEBUG == 0) return ; 
    std::cout <<"Tensor "<< name << " row:cols " << rows << ":"<< cols << " created " << std::endl;
}

unsigned int Tensor::rows() const{ return m_rows; }
unsigned int Tensor::cols() const{ return m_cols; }
float* Tensor::data() const{ return m_data; }

void Tensor::checkInBounds(const Tensor &A, unsigned int x, unsigned int y){
    if(x >= A.m_rows || y >= A.m_cols) {
        std::cout << "Error: Index out of bounds."<< std::endl;
        throw 0;
    }
}

void Tensor::checkEqualSize(const Tensor &A, const Tensor &B){
    if(A.m_rows != B.m_rows || A.m_cols != B.m_cols) {
        std::cout << "Error: " << "Tensors must have the same size" << std::endl;
        throw 0;
    }
}

void Tensor::checkMatMulPossible(const Tensor &A, const Tensor &B){
    if(A.m_cols != B.m_rows)
    {
        std::cout << "MatMult size error: A.cols() = " << A.m_cols
        << " != B.rows() = "<< B.m_rows << " " << std::endl;
        throw 0;
    }
}

Tensor Tensor::copy(){
    Tensor copy = Tensor(m_rows,m_cols,m_data);
    return copy;
}

Tensor Tensor::relu(){
    Tensor  rl = Tensor(m_rows, m_cols);
    float * input = m_data;
    int input_len = m_rows * m_cols;
    for (int i = 0; i < input_len; i++)  rl[i] = input[i] >0 ?  input[i] : 0 ;
    return rl;
}

Tensor Tensor::soft_max(){
    Tensor sm  = Tensor(m_rows, m_cols);
    float * input = m_data;
    int input_len = m_rows * m_cols;
    assert (input != NULL);
    assert (input_len != 0);
    int i;
    float max;
    max = input[0]; 
    for (i = 1; i < input_len; i++)   if (input[i] > max)  max = input[i];
    float sum = 0.0;
    for (i = 0; i < input_len; i++)  sum += expf(input[i]-max);
    for (i = 0; i < input_len; i++)   sm[i] = input[i] = expf(input[i] - max - log(sum));
    return sm;
}

Tensor Tensor::matmul(const Tensor &A, const Tensor &B){ // Static methods
    checkMatMulPossible(A,B);
    unsigned int N = A.m_rows;
    unsigned int K = A.m_cols;
    unsigned int M = B.m_cols;  //   printf( "(%d) A N%d:K%d B %d:M%d\n",__LINE__, N, K,B.m_rows, M );
    std::string na = " # " + A.name + B.name ;
    Tensor C = Tensor(N,M, nullptr, na );
    int ctr =1;
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
           for(int k = 0; k < K; ++k)  C(i,j) += (float) A(i,k)*B(k,j);
        }
    }
    return C;
}

float& Tensor::operator[](unsigned int index){  return m_data[index]; } // Operators

float& Tensor::operator()(unsigned int x, unsigned int y) const {
    checkInBounds(*this,x,y);
    return m_data[x*m_cols+y];
}

Tensor Tensor::operator+(const Tensor &other){
    checkEqualSize(*this,other);
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){ result.m_data[i] += other.m_data[i]; }
    return result;
}

Tensor Tensor::operator+(const float c){
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){ result.m_data[i] += c; }
    return result;
}

Tensor Tensor::operator*(const Tensor &other){
    checkEqualSize(*this,other);
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){ result.m_data[i] *= other.m_data[i]; }
    return result;
}

Tensor Tensor::operator*(const float c){
    Tensor result = this->copy();
    for(int i = 0; i < m_rows*m_cols; i++){ result.m_data[i] *= c; }
    return result;
}

std::string Tensor::toString(){
    if (DEBUG == 0) return ""; 
    std::stringstream ss;
    ss << " row=" << m_rows << " col=" << m_cols << "\n";
    for(int i = 0; i < m_rows; i++){
       for(int j = 0; j < m_cols; j++)
            ss  << m_data[i*m_cols+j] << " | ";
       ss << std::endl;
    }
    return ss.str();
}

void  Tensor::print(){
    if (DEBUG == 0) return; 
    int len = m_rows * m_cols;
    float * p = (float*) m_data;
    for( int i=0; i<5; i++ ) { printf( "%+10.3f, ", p[i] ); } printf( " ... ");
    for( int i=len-5; i<len; i++ ){ printf( "%+10.3f, ", p[i] ); } printf( "\n");
}
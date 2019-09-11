#include </users/henney/Documents/Oxford/C++_libraries/pybind11/include/pybind11/pybind11.h>
#include </users/henney/Documents/Oxford/C++_libraries/pybind11/include/pybind11/stl.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <math.h>
namespace py = pybind11;
using namespace std;
template <typename V>
V get(py::dict m, const std::string &key, const V &defval) {
    cout<<defval<<"\n";
    return m[key.c_str()].cast<V>();
}
double mean(vector<double> data){
  double mean_val=0;
  int length=data.size();
  for (int i=0;  i<length; i++) {
    mean_val=mean_val+data[i];
  }
  return mean_val/data.size();
}
double stdev(vector<double> data){
  double sample_mean=mean(data);
  double summation=0;
  int length=data.size();
  for(int i=0;  i<length; i++){
    summation=summation+pow((data[i]-sample_mean),2);
  }
  double std=sqrt(summation/data.size());
  return std;
}
vector<vector<vector<double>>> white_scanner(const vector<vector<vector<double>>> image,double threshold){
  int rows=image.size();
  int cols=image[0].size();
  int num_pixels=image[0][0].size();
  vector<vector<vector<double>>> processed_image;
  processed_image.resize(rows);
  for(int i = 0; i < rows; i++){
    processed_image[i].resize(cols);
    for(int j = 0; j < cols; j++){
        processed_image[i][j].resize(num_pixels);
      }
    }
  vector<double> RGB_pix(3,0);
  double std;
  for(int i=0;  i<rows; i++){
    for(int j=0;  j<cols; j++){
      for (int k=0;  k<num_pixels; k++){
        RGB_pix[k]=image[i][j][k];
      }
      std=stdev(RGB_pix);
      if (std<=threshold){
        for (int k=0;  k<num_pixels; k++){
          processed_image[i][j][k]=image[i][j][k];
        }
      }else{
        for (int k=0;  k<num_pixels; k++){
          processed_image[i][j][k]=0;
        }
      }
    }
  }
  return processed_image;
}

int main () {
  return 0;
}
PYBIND11_MODULE(image_funcs, m) {
  m.def("mean", &mean, "calculate sample mean");
  m.def("stdev", &stdev, "calculate sample standard deviation");
  m.def("white_scanner", &white_scanner, "Sets pixels with large standard deviations to 0");
}

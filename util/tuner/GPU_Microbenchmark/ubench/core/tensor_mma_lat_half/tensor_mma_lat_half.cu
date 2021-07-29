#include "tensor_mma_lat_half.h"


int main() {

  intilizeDeviceProp(0);

  if (deviceProp.major < 8) // tesnore unit was added since Volta
    return 1;

  std::cout << "FP16 operand, FP32 accumalte:\n";
  tensor_lat<half, float>();

  std::cout << "\nFP16 operand, FP16 accumalte:\n";
  tensor_lat<half, half>();

  std::cout << "\nmma1688 FP16 operand, FP32 accumalte:\n";
  tensor1688_lat<half, float>();

  std::cout << "\nmma1688 FP16 operand, FP16 accumalte:\n";
  tensor1688_lat<half, half>();


  
  // tensor_lat<char,int>();

  return 1;
}

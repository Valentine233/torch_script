#include <torch/torch.h>
#include "flash_api.cpp"

#include <iostream>

int main() {
    at::Device device = at::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        device = at::kCUDA;
    } else {
        std::cout << "CUDA is not available." << std::endl;
    }

    // Define the sizes for the tensors
    int batch_size = 2;
    int seqlen_q = 128;
    int seqlen_k = 128;
    int num_heads = 5;
    int head_size = 64;

    // Create random tensors for q, k, v
    at::Tensor q = at::randn({batch_size, seqlen_q, num_heads, head_size}, device).to(at::kBFloat16);
    at::Tensor k = at::randn({batch_size, seqlen_k, num_heads, head_size}, device).to(at::kBFloat16);
    at::Tensor v = at::randn({batch_size, seqlen_k, num_heads, head_size}, device).to(at::kBFloat16);

    // Optional parameters
    c10::optional<at::Tensor> out_ = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes_ = c10::nullopt;
    c10::optional<at::Generator> gen_ = c10::nullopt;

    // Call the mha_fwd function
    std::vector<at::Tensor> result = mha_fwd(
        q, k, v, out_, alibi_slopes_, 
        0.0f, // dropout
        1.0f, // softmax scale
        true, // is causal
        seqlen_q,    // window size left
        seqlen_q,    // window size right
        0.0f, // softcap
        false, // return softmax
        gen_
    );

    // Output the result
    std::cout << "Output Tensor: " << result[0] << std::endl;

    return 0;
}

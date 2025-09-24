#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

torch::Tensor fused_bias_leakyrelu_op(const torch::Tensor &input,
                                      const torch::Tensor &bias,
                                      const torch::Tensor &refer, int act,
                                      int grad, float alpha, float scale);

torch::Tensor fused_bias_leakyrelu_op_impl(const torch::Tensor &input,
                                           const torch::Tensor &bias,
                                           const torch::Tensor &refer, int act,
                                           int grad, float alpha, float scale);
REGISTER_DEVICE_IMPL(fused_bias_leakyrelu_op_impl, CUDA,
                     fused_bias_leakyrelu_op);

torch::Tensor bias_act_op_impl(const torch::Tensor &input,
                               const torch::Tensor &bias,
                               const torch::Tensor &xref,
                               const torch::Tensor &yref,
                               const torch::Tensor &dy, int grad, int dim,
                               int act, float alpha, float gain, float clamp);

torch::Tensor bias_act_op(const torch::Tensor &input, const torch::Tensor &bias,
                          const torch::Tensor &xref, const torch::Tensor &yref,
                          const torch::Tensor &dy, int grad, int dim, int act,
                          float alpha, float gain, float clamp);

REGISTER_DEVICE_IMPL(bias_act_op_impl, CUDA, bias_act_op);

torch::Tensor filtered_lrelu_act_op_impl(torch::Tensor x, torch::Tensor si,
                                         int sx, int sy, float gain,
                                         float slope, float clamp,
                                         bool writeSigns);

torch::Tensor filtered_lrelu_act_op(torch::Tensor x, torch::Tensor si, int sx,
                                    int sy, float gain, float slope,
                                    float clamp, bool writeSigns);

REGISTER_DEVICE_IMPL(filtered_lrelu_act_op_impl, CUDA, filtered_lrelu_act_op);

torch::Tensor upfirdn2d_op(torch::Tensor input, torch::Tensor filter, int upx,
                           int upy, int downx, int downy, int padx0, int padx1,
                           int pady0, int pady1, bool flip, float gain);

torch::Tensor upfirdn2d_op_impl(torch::Tensor input, torch::Tensor filter,
                                int upx, int upy, int downx, int downy,
                                int padx0, int padx1, int pady0, int pady1,
                                bool flip, float gain);
REGISTER_DEVICE_IMPL(upfirdn2d_op_impl, CUDA, upfirdn2d_op);

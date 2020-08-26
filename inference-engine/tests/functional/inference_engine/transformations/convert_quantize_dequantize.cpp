// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/convert_quantize_dequantize.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


template <typename T>
std::shared_ptr<Function> create_q_dq_function(const Shape& data_shape, float in_low, float in_high, float out_low, float out_high,
                                               const Shape& zero_point_shape, std::vector<T> zero_point_values,
                                               const Shape& scale_shape, std::vector<float> scale_values, size_t levels) {
    auto data = std::make_shared<opset1::Parameter>(element::f32, data_shape);
    auto input_low = opset1::Constant::create(element::f32, Shape{}, {in_low});
    auto input_high = opset1::Constant::create(element::f32, Shape{}, {in_high});
    auto output_low = opset1::Constant::create(element::f32, Shape{}, {out_low});
    auto output_high = opset1::Constant::create(element::f32, Shape{}, {out_high});
    auto fq = std::make_shared<opset1::FakeQuantize>(data, input_low,
                                                     input_high, output_low,
                                                     output_high, levels);
    auto convert1 = std::make_shared<opset1::Convert>(fq, element::from<T>());
    auto convert2 = std::make_shared<opset1::Convert>(convert1, element::f32);
    auto zero_point = std::make_shared<opset1::Convert>(opset1::Constant::create(element::from<T>(), zero_point_shape, zero_point_values), element::f32);
    auto sub = std::make_shared<opset1::Subtract>(convert2, zero_point);
    auto scale = opset1::Constant::create(element::f32, scale_shape, scale_values);
    auto mul = std::make_shared<opset1::Multiply>(sub, scale);

    return std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});
}

template <typename T>
void positive_test(const Shape& data_shape, float in_low, float in_high, float out_low, float out_high,
                   const Shape& zero_point_shape, std::vector<T> zero_point_values,
                   const Shape& scale_shape, std::vector<float> scale_values, size_t levels) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        f = create_q_dq_function(data_shape, in_low, in_high, out_low, out_high,
                                 zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertQuantizeDequantize>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, data_shape);
        auto input_low = opset1::Constant::create(element::f32, Shape{}, {in_low});
        auto input_high = opset1::Constant::create(element::f32, Shape{}, {in_high});
        auto output_low = opset1::Constant::create(element::f32, Shape{}, {(out_low - zero_point_values[0]) * scale_values[0]});
        auto output_high = opset1::Constant::create(element::f32, Shape{}, {(out_high - zero_point_values[0]) * scale_values[0]});
        auto fq = std::make_shared<opset1::FakeQuantize>(data, input_low,
                                                         input_high, output_low,
                                                         output_high, levels);
        f_ref = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertQuantizeDequantizeINT8) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    Shape data_shape{3, 1, 2};
    float in_low = 0;
    float in_high = 5;
    float out_low = -128;
    float out_high = 127;
    Shape zero_point_shape{};
    std::vector<int8_t> zero_point_values{2};
    Shape scale_shape{};
    std::vector<float> scale_values{3};
    size_t levels = 256;

    positive_test(data_shape, in_low, in_high, out_low, out_high,
                  zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
}

TEST(TransformationTests, ConvertQuantizeDequantizeUINT8) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    Shape data_shape{3, 1, 2};
    float in_low = 0;
    float in_high = 5;
    float out_low = 0;
    float out_high = 255;
    Shape zero_point_shape{};
    std::vector<uint8_t> zero_point_values{2};
    Shape scale_shape{};
    std::vector<float> scale_values{3};
    size_t levels = 256;

    positive_test(data_shape, in_low, in_high, out_low, out_high,
                  zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
}

template <typename T>
void negative_test(const Shape& data_shape, float in_low, float in_high, float out_low, float out_high,
                   const Shape& zero_point_shape, std::vector<T> zero_point_values,
                   const Shape& scale_shape, std::vector<float> scale_values, size_t levels) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        f = create_q_dq_function(data_shape, in_low, in_high, out_low, out_high,
                                 zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertQuantizeDequantize>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        // negative test so the transformation does not fire and reference is the same graph as original
        f_ref = create_q_dq_function(data_shape, in_low, in_high, out_low, out_high,
                                 zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertQuantizeDequantizeZeroPointNotBroadcastable) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    Shape data_shape{3, 1, 2};
    float in_low = 0;
    float in_high = 5;
    float out_low = -128;
    float out_high = 127;
    Shape zero_point_shape{1, 1, 1, 1};
    std::vector<int8_t> zero_point_values{2};
    Shape scale_shape{1};
    std::vector<float> scale_values{3};
    size_t levels = 256;

    negative_test(data_shape, in_low, in_high, out_low, out_high,
                  zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
}

TEST(TransformationTests, ConvertQuantizeDequantizeScaleNotBroadcastable) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    Shape data_shape{3, 1, 2};
    float in_low = 0;
    float in_high = 5;
    float out_low = -128;
    float out_high = 127;
    Shape zero_point_shape{};
    std::vector<int8_t> zero_point_values{2};
    Shape scale_shape{1, 1, 1, 1};
    std::vector<float> scale_values{3};
    size_t levels = 256;

    negative_test(data_shape, in_low, in_high, out_low, out_high,
                  zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
}

TEST(TransformationTests, ConvertQuantizeDequantizeInvalidLevels) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    Shape data_shape{3, 1, 2};
    float in_low = 0;
    float in_high = 5;
    float out_low = -128;
    float out_high = 127;
    Shape zero_point_shape{};
    std::vector<int8_t> zero_point_values{2};
    Shape scale_shape{};
    std::vector<float> scale_values{3};
    size_t levels = 127;

    negative_test(data_shape, in_low, in_high, out_low, out_high,
                  zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
}

TEST(TransformationTests, ConvertQuantizeDequantizeInvalidOutLowOutHigh) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    Shape data_shape{3, 1, 2};
    float in_low = 0;
    float in_high = 5;
    // (-128, 127) are invalid for uin8_t data type
    float out_low = -128;
    float out_high = 127;
    Shape zero_point_shape{};
    std::vector<uint8_t> zero_point_values{2};
    Shape scale_shape{};
    std::vector<float> scale_values{3};
    size_t levels = 256;

    negative_test(data_shape, in_low, in_high, out_low, out_high,
                  zero_point_shape, zero_point_values, scale_shape, scale_values, levels);
}

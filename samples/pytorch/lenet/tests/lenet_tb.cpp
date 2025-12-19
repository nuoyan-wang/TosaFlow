#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <iostream>
#include "accelerator_hls.h"


int main() {
    const std::string label_path  = "C:/Users/binhm/Documents/School/ece527/ece527-bmn4-nuoyanw2/mp4/lenet_files/labels.bin";
    const std::string image_path  = "C:/Users/binhm/Documents/School/ece527/ece527-bmn4-nuoyanw2/mp4/lenet_files/images.bin";
    #ifdef D_INT
    const std::string param_path  = "C:/Users/binhm/Documents/School/ece527/ece527-bmn4-nuoyanw2/mp4/params.bin";
    #else
    const std::string param_path  = "C:/Users/binhm/Documents/School/ece527/ece527-bmn4-nuoyanw2/mp4/lenet_files/params.bin";
    #endif

    const unsigned num_images     = 10;
    const unsigned batch_size     = 1;
    const unsigned input_channels = 1;
    const unsigned input_h        = 32;
    const unsigned input_w        = 32;

     // Load parameters
    std::ifstream param_file(param_path, std::ios::binary);
    assert(param_file.is_open());
    param_file.seekg(0, std::ios::end);
    size_t param_size = param_file.tellg();
    param_file.seekg(0, std::ios::beg);
    std::vector<dataType> params(param_size / sizeof(dataType));
    param_file.read(reinterpret_cast<char*>(params.data()), param_size);
    param_file.close();

    size_t idx = 0;
    auto slice = [&](size_t n) { 
        std::vector<dataType> v(params.begin() + idx, params.begin() + idx + n); 
        idx += n;
        return v;
    };

    std::vector<dataType> conv1_w = slice(6*1*5*5);
    std::vector<dataType> conv1_b = slice(6);
    std::vector<dataType> conv2_w = slice(16*6*5*5);
    std::vector<dataType> conv2_b = slice(16);
    std::vector<dataType> conv3_w = slice(120*16*5*5);
    std::vector<dataType> conv3_b = slice(120);
    std::vector<dataType> fc_w    = slice(10*120);
    std::vector<dataType> fc_b    = slice(10);

    std::cout << "idx/params.size: " << idx << "/" << params.size() << std::endl;

    assert(idx == params.size());

    // Load labels
    std::ifstream f_label(label_path, std::ios::binary);
    assert(f_label.is_open());
    uint32_t magic, num_labels;
    f_label.read(reinterpret_cast<char*>(&magic), 4);
    f_label.read(reinterpret_cast<char*>(&num_labels), 4);
    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);
    std::vector<uint8_t> labels(num_labels);
    f_label.read(reinterpret_cast<char*>(labels.data()), num_labels);
    f_label.close();

    // Load images
    std::ifstream f_img(image_path, std::ios::binary);
    assert(f_img.is_open());
    uint32_t magic_i, num_img, rows, cols;
    f_img.read(reinterpret_cast<char*>(&magic_i), 4);
    f_img.read(reinterpret_cast<char*>(&num_img), 4);
    f_img.read(reinterpret_cast<char*>(&rows), 4);
    f_img.read(reinterpret_cast<char*>(&cols), 4);
    magic_i = __builtin_bswap32(magic_i);
    num_img = __builtin_bswap32(num_img);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<uint8_t> pixels(num_img * rows * cols);
    f_img.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    f_img.close();

    // Preprocess images: (28x28) -> (1x32x32) float [-1,1]
    #ifdef D_INT
    std::vector<dataType> input_buffer(num_images * input_channels * input_h * input_w, -128);
    #else
    std::vector<dataType> input_buffer(num_images * input_channels * input_h * input_w, -1.0f);
    #endif

    for (size_t n = 0; n < num_images; ++n) {
        for (size_t ic = 0; ic < input_channels; ++ic) {
            for (size_t y = 0; y < input_h; ++y) {
                for (size_t x = 0; x < input_w; ++x) {
                    size_t dst_idx = ((n * input_channels + ic) * input_h + y) * input_w + x;

                    // 2px border = -1, core pixels scaled to [-1,1]
                    if (y >= 2 && y < 30 && x >= 2 && x < 30) {
                        size_t src_idx = n * 28 * 28 + (y - 2) * 28 + (x - 2);
                        #ifdef D_INT
                        input_buffer[dst_idx] = reinterpret_cast<dataType&>(pixels[src_idx]) - 128;
                        #else
                        input_buffer[dst_idx] = static_cast<dataType>(pixels[src_idx]) / 255.0f * 2.0f - 1.0f;
                        #endif
                    }
                }
            }
        }
    }

    // Allocate output
    // std::vector<dataType> output_buffer(num_images*10, 0);
    std::vector<dataType> output_buffer(num_images*14*14*6, 0);

    static float input_img[32][32];
    static float output_logits[10];

    static float W3[5][5][16][120];
    static float W2[5][5][6][16];
    static float Wfc[120][10];

    static float buf0[28][28][6];
    static float buf1[28][28][6];
    static float buf2[28][28][6];
    static float buf3[28][28][6];

    static float pool0[14][14][6];
    static float pool1[14][14][6];
    static float pool2[14][14][6];

    static float conv2a[10][10][16];
    static float conv2b[10][10][16];
    static float conv2c[10][10][16];
    static float conv2d[10][10][16];

    memcpy(W3, conv3_w.data(), sizeof(W3));
    memcpy(W2, conv2_w.data(), sizeof(W2));
    memcpy(Wfc, fc_w.data(), sizeof(Wfc));


    // lenet_top
    for (unsigned start = 0; start < num_images; start += batch_size) {
        printf("[INFO] Starting batch starting at image %u\n", start);
        unsigned this_batch = std::min(batch_size, num_images - start);
        memcpy(input_img, &input_buffer[start * input_h * input_w], sizeof(input_img));

        forward_lenet(
            input_img,
            output_logits,
            W3,
            W2,
            Wfc,
            buf0, buf1, buf2, buf3,
            pool0, pool1, pool2,
            conv2a, conv2b, conv2c, conv2d
        );

        memcpy(&output_buffer[start * 10], output_logits, sizeof(output_logits));
    }

    // Evaluate accuracy
    unsigned correct = 0;
    for (unsigned n = 0; n < num_images; ++n) {
        int pred = 0;
        float max_val = output_buffer[n*10];
        // if (n == 0) {
        //     printf("Output for image %u:\n", n);
        //     for (int k = 0; k < 10; ++k) {
        //         printf("  class %d: %.6f\n", k, output_buffer[n * 10 + k]);
        //     }
        // }

        for (int k = 1; k < 10; ++k) {
            if (output_buffer[n*10 + k] > max_val) {
                max_val = output_buffer[n*10 + k];
                pred = k;
            }
        }
        if (pred == labels[n]) correct++;
    }

    for(int j = 0; j < batch_size; ++j) {
        std::cout << "outputbuf " << j << ": ";
        for(int i = 0; i < 10; ++i) {
            std::cout << +output_buffer[10 * j + i] << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "C-sim accuracy: " << (100.0*correct/num_images) << "% (" << correct << "/" << num_images << ")\n";

    // Post Pool1
    // std::cout << "outputbuf: ";
    // for(int oc = 0; oc < 6; ++oc) {
    //     for(int y = 0; y < 14; ++y) {
    //         for(int x = 0; x < 14; ++x) {
    //             std::cout << +output_buffer[6 * 14 * y + 6 * x + oc] << ", ";
    //         }
    //     }
    // }
    // std::cout << std::endl;

    // Post Pool2
    // std::cout << "outputbuf: ";
    // for(int oc = 0; oc < 16; ++oc) {
    //     for(int y = 0; y < 5; ++y) {
    //         for(int x = 0; x < 5; ++x) {
    //             std::cout << +output_buffer[16 * 5 * y + 16 * x + oc] << ", ";
    //         }
    //     }
    // }
    // std::cout << std::endl;


    return 0;
}